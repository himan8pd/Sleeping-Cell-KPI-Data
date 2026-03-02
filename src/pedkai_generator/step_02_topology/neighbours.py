"""
Cell neighbour relations builder.

Generates cell-to-cell adjacency / neighbour relationships with handover metrics.

Neighbour types:
  - INTRA_FREQ_NEIGHBOUR: same band, same RAT
  - INTER_FREQ_NEIGHBOUR: different band, same RAT
  - INTER_RAT_NEIGHBOUR: LTE ↔ NR for NSA handover

Each neighbour relation has:
  - handover_attempts (measured over simulation period)
  - handover_success_rate (%)
  - cio_offset_db (Cell Individual Offset — configurable)
  - no_remove_flag (operator-locked neighbour)

Target: ~200K neighbour pairs.

Output feeds into: neighbour_relations.parquet
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console

from pedkai_generator.config.settings import GeneratorConfig
from pedkai_generator.step_02_topology.builders import (
    TopologyAccumulator,
    haversine_distance_m,
    make_neighbour_relation,
    make_relationship,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum distance (metres) for neighbour consideration per deployment profile
MAX_NEIGHBOUR_DISTANCE_M: dict[str, float] = {
    "dense_urban": 800.0,
    "urban": 1500.0,
    "suburban": 4000.0,
    "rural": 12000.0,
    "deep_rural": 30000.0,
    "indoor": 500.0,
}

# Maximum number of neighbours per cell (to keep the graph manageable)
MAX_NEIGHBOURS_PER_CELL: int = 32

# Intra-site neighbours are always added (cells on the same site)
# Inter-site neighbours are distance-gated

# Handover success rate distributions by neighbour type
HO_SUCCESS_RATE_PARAMS: dict[str, tuple[float, float]] = {
    # (mean, std) — truncated to [0, 100]
    "INTRA_FREQ_NEIGHBOUR": (98.5, 1.0),
    "INTER_FREQ_NEIGHBOUR": (96.5, 2.0),
    "INTER_RAT_NEIGHBOUR": (93.0, 3.5),
}

# Handover attempt count distributions by neighbour type
# (attempts over 30-day simulation period)
HO_ATTEMPT_PARAMS: dict[str, tuple[float, float]] = {
    # (mean, std) — lognormal parameters
    "INTRA_FREQ_NEIGHBOUR": (6.0, 1.0),  # ~400 attempts mean
    "INTER_FREQ_NEIGHBOUR": (5.0, 1.2),  # ~150 attempts mean
    "INTER_RAT_NEIGHBOUR": (4.5, 1.5),  # ~90 attempts mean
}


# ---------------------------------------------------------------------------
# Spatial index: simple grid-based approach
# ---------------------------------------------------------------------------


class SpatialGrid:
    """
    A simple spatial grid index for fast nearest-neighbour lookups.

    Divides the geographic area into grid cells of a fixed size and
    allows querying for all cells within a certain distance of a point.
    """

    def __init__(self, grid_size_deg: float = 0.1):
        """
        Args:
            grid_size_deg: Grid cell size in degrees. 0.1° ≈ 11 km at equator.
        """
        self.grid_size = grid_size_deg
        self.grid: dict[tuple[int, int], list[dict[str, Any]]] = {}

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(lat / self.grid_size), int(lon / self.grid_size))

    def insert(self, cell: dict[str, Any]) -> None:
        lat = cell.get("geo_lat") or 0.0
        lon = cell.get("geo_lon") or 0.0
        key = self._key(lat, lon)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(cell)

    def query_nearby(self, lat: float, lon: float, radius_deg: float) -> list[dict[str, Any]]:
        """Return all cells within radius_deg of the given point."""
        results: list[dict[str, Any]] = []
        n_cells = int(radius_deg / self.grid_size) + 1

        center_key = self._key(lat, lon)
        for di in range(-n_cells, n_cells + 1):
            for dj in range(-n_cells, n_cells + 1):
                key = (center_key[0] + di, center_key[1] + dj)
                if key in self.grid:
                    results.extend(self.grid[key])
        return results


# ---------------------------------------------------------------------------
# Neighbour type determination
# ---------------------------------------------------------------------------


def _determine_neighbour_type(
    from_cell: dict[str, Any],
    to_cell: dict[str, Any],
) -> str | None:
    """
    Determine the neighbour relation type between two cells.

    Returns None if the cells should not be neighbours (e.g., same cell,
    or an NR SCG leg trying to neighbour another NR SCG leg on the same site
    without going through the anchor).
    """
    from_rat = from_cell.get("rat_type", "")
    to_rat = to_cell.get("rat_type", "")
    from_band = from_cell.get("band", "")
    to_band = to_cell.get("band", "")

    # Same cell
    if from_cell["entity_id"] == to_cell["entity_id"]:
        return None

    # Both LTE
    if from_rat in ("LTE", "NR_NSA") and to_rat in ("LTE", "NR_NSA"):
        # Check if both are the LTE-layer (LTE_CELL entity type)
        if from_cell["entity_type"] == "LTE_CELL" and to_cell["entity_type"] == "LTE_CELL":
            if from_band == to_band:
                return "INTRA_FREQ_NEIGHBOUR"
            else:
                return "INTER_FREQ_NEIGHBOUR"

    # Both NR (SA or NSA SCG legs)
    if from_cell["entity_type"] == "NR_CELL" and to_cell["entity_type"] == "NR_CELL":
        if from_band == to_band:
            return "INTRA_FREQ_NEIGHBOUR"
        else:
            return "INTER_FREQ_NEIGHBOUR"

    # Inter-RAT: LTE ↔ NR (for NSA handover or SRVCC-like scenarios)
    if (from_cell["entity_type"] == "LTE_CELL" and to_cell["entity_type"] == "NR_CELL") or (
        from_cell["entity_type"] == "NR_CELL" and to_cell["entity_type"] == "LTE_CELL"
    ):
        return "INTER_RAT_NEIGHBOUR"

    return None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_neighbour_relations(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Build cell neighbour relations.

    Strategy:
    1. Index all cells by geographic location using a spatial grid.
    2. For each cell, find nearby cells and establish neighbour relations.
    3. Intra-site neighbours are always created (cells on same physical site).
    4. Inter-site neighbours are distance-gated based on deployment profile.
    5. Each relation gets handover metrics and CIO offset.

    All neighbour relations are added to the accumulator and also create
    NEIGHBOURS relationships in the topology graph.

    Target: ~200K neighbour pairs.
    """
    tenant = config.tenant_id

    # Gather all cell entities (LTE_CELL and NR_CELL)
    lte_cells = acc.get_entities_by_type("LTE_CELL")
    nr_cells = acc.get_entities_by_type("NR_CELL")
    all_cells = lte_cells + nr_cells

    if not all_cells:
        console.print("    [yellow]No cells found — skipping neighbour relations.[/yellow]")
        return

    console.print(
        f"    [dim]Building neighbours for {len(all_cells):,} cells "
        f"(LTE: {len(lte_cells):,}, NR: {len(nr_cells):,})...[/dim]"
    )

    # ── Build spatial index ──────────────────────────────────
    # Use a grid size appropriate for the densest deployment (~0.005° ≈ 500m)
    grid = SpatialGrid(grid_size_deg=0.02)  # ~2.2 km grid cells
    for cell in all_cells:
        grid.insert(cell)

    # ── Group cells by site for intra-site neighbour detection ─
    cells_by_site: dict[str, list[dict[str, Any]]] = {}
    for cell in all_cells:
        sid = cell.get("site_id", "")
        if sid:
            if sid not in cells_by_site:
                cells_by_site[sid] = []
            cells_by_site[sid].append(cell)

    # ── Build neighbour pairs ────────────────────────────────
    # Use a set to track unique pairs (unordered) to avoid duplicates
    seen_pairs: set[tuple[str, str]] = set()
    neighbour_rows: list[dict[str, Any]] = []
    topology_rels: list[dict[str, Any]] = []

    n_intra_site = 0
    n_inter_site = 0
    n_inter_rat = 0

    # Process cells in chunks to manage memory
    for cell_idx, cell in enumerate(all_cells):
        cell_id = cell["entity_id"]
        cell_lat = cell.get("geo_lat") or 0.0
        cell_lon = cell.get("geo_lon") or 0.0
        cell_site = cell.get("site_id", "")
        cell_profile = cell.get("deployment_profile", "suburban")

        max_dist_m = MAX_NEIGHBOUR_DISTANCE_M.get(cell_profile, 4000.0)
        # Convert to approximate degrees for spatial query
        max_dist_deg = max_dist_m / 111_000.0  # ~111km per degree

        neighbours_added = 0

        # ── 1. Intra-site neighbours (always add) ────────────
        if cell_site and cell_site in cells_by_site:
            for other in cells_by_site[cell_site]:
                other_id = other["entity_id"]
                if other_id == cell_id:
                    continue

                pair_key = (min(cell_id, other_id), max(cell_id, other_id))
                if pair_key in seen_pairs:
                    continue

                ntype = _determine_neighbour_type(cell, other)
                if ntype is None:
                    continue

                seen_pairs.add(pair_key)

                # Generate handover metrics
                ho_success_mean, ho_success_std = HO_SUCCESS_RATE_PARAMS[ntype]
                ho_success = float(np.clip(rng.normal(ho_success_mean, ho_success_std), 50.0, 100.0))

                ho_att_mu, ho_att_sigma = HO_ATTEMPT_PARAMS[ntype]
                ho_attempts = max(1.0, float(rng.lognormal(ho_att_mu, ho_att_sigma)))

                # Intra-site: higher attempts, better success
                ho_attempts *= 1.5
                ho_success = min(100.0, ho_success + 0.5)

                cio = round(float(rng.normal(0, 2.0)), 1)
                cio = float(np.clip(cio, -24.0, 24.0))

                no_remove = bool(rng.random() < 0.05)  # 5% operator-locked

                nr = make_neighbour_relation(
                    tenant_id=tenant,
                    from_cell_id=cell_id,
                    from_cell_rat=cell.get("rat_type", "LTE"),
                    from_cell_band=cell.get("band", ""),
                    to_cell_id=other_id,
                    to_cell_rat=other.get("rat_type", "LTE"),
                    to_cell_band=other.get("band", ""),
                    neighbour_type=ntype,
                    is_intra_site=True,
                    distance_m=0.0,  # Same site
                    handover_attempts=round(ho_attempts, 0),
                    handover_success_rate=round(ho_success, 2),
                    cio_offset_db=cio,
                    no_remove_flag=no_remove,
                )
                neighbour_rows.append(nr)
                neighbours_added += 1
                n_intra_site += 1

                if ntype == "INTER_RAT_NEIGHBOUR":
                    n_inter_rat += 1

                # Also create a NEIGHBOURS relationship in the topology graph
                topology_rels.append(
                    make_relationship(
                        tenant_id=tenant,
                        from_entity_id=cell_id,
                        from_entity_type=cell["entity_type"],
                        relationship_type="NEIGHBOURS",
                        to_entity_id=other_id,
                        to_entity_type=other["entity_type"],
                        domain="mobile_ran",
                    )
                )

        # ── 2. Inter-site neighbours (distance-gated) ────────
        if neighbours_added >= MAX_NEIGHBOURS_PER_CELL:
            continue

        nearby = grid.query_nearby(cell_lat, cell_lon, max_dist_deg)

        # Sort by distance for priority
        candidates: list[tuple[float, dict[str, Any]]] = []
        for other in nearby:
            other_id = other["entity_id"]
            if other_id == cell_id:
                continue
            other_site = other.get("site_id", "")
            if other_site == cell_site:
                continue  # Already handled as intra-site

            pair_key = (min(cell_id, other_id), max(cell_id, other_id))
            if pair_key in seen_pairs:
                continue

            other_lat = other.get("geo_lat") or 0.0
            other_lon = other.get("geo_lon") or 0.0
            dist = haversine_distance_m(cell_lat, cell_lon, other_lat, other_lon)

            if dist > max_dist_m:
                continue

            candidates.append((dist, other))

        # Sort by distance, take closest
        candidates.sort(key=lambda x: x[0])
        max_inter = MAX_NEIGHBOURS_PER_CELL - neighbours_added

        for dist, other in candidates[:max_inter]:
            other_id = other["entity_id"]
            pair_key = (min(cell_id, other_id), max(cell_id, other_id))
            if pair_key in seen_pairs:
                continue

            ntype = _determine_neighbour_type(cell, other)
            if ntype is None:
                continue

            seen_pairs.add(pair_key)

            # Generate handover metrics — scale with distance
            ho_success_mean, ho_success_std = HO_SUCCESS_RATE_PARAMS[ntype]
            # Farther = slightly lower success rate
            dist_penalty = min(5.0, dist / max_dist_m * 5.0)
            ho_success = float(
                np.clip(
                    rng.normal(ho_success_mean - dist_penalty, ho_success_std + 0.5),
                    50.0,
                    100.0,
                )
            )

            ho_att_mu, ho_att_sigma = HO_ATTEMPT_PARAMS[ntype]
            # Farther = fewer attempts (less overlap)
            dist_factor = max(0.1, 1.0 - (dist / max_dist_m) * 0.7)
            ho_attempts = max(1.0, float(rng.lognormal(ho_att_mu, ho_att_sigma)) * dist_factor)

            cio = round(float(rng.normal(0, 3.0)), 1)
            cio = float(np.clip(cio, -24.0, 24.0))

            no_remove = bool(rng.random() < 0.02)  # 2% operator-locked for inter-site

            nr = make_neighbour_relation(
                tenant_id=tenant,
                from_cell_id=cell_id,
                from_cell_rat=cell.get("rat_type", "LTE"),
                from_cell_band=cell.get("band", ""),
                to_cell_id=other_id,
                to_cell_rat=other.get("rat_type", "LTE"),
                to_cell_band=other.get("band", ""),
                neighbour_type=ntype,
                is_intra_site=False,
                distance_m=round(dist, 1),
                handover_attempts=round(ho_attempts, 0),
                handover_success_rate=round(ho_success, 2),
                cio_offset_db=cio,
                no_remove_flag=no_remove,
            )
            neighbour_rows.append(nr)
            neighbours_added += 1
            n_inter_site += 1

            if ntype == "INTER_RAT_NEIGHBOUR":
                n_inter_rat += 1

            topology_rels.append(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=cell_id,
                    from_entity_type=cell["entity_type"],
                    relationship_type="NEIGHBOURS",
                    to_entity_id=other_id,
                    to_entity_type=other["entity_type"],
                    domain="mobile_ran",
                )
            )

        # Progress logging every 10K cells
        if (cell_idx + 1) % 10000 == 0:
            console.print(
                f"      [dim]Processed {cell_idx + 1:,}/{len(all_cells):,} cells, "
                f"{len(neighbour_rows):,} pairs so far...[/dim]"
            )

    # ── Add to accumulator ───────────────────────────────────
    acc.add_neighbour_relations(neighbour_rows)
    acc.add_relationships(topology_rels)

    # ── Summary ──────────────────────────────────────────────
    total_pairs = len(neighbour_rows)

    # Count by type
    type_counts: dict[str, int] = {}
    for nr in neighbour_rows:
        nt = nr.get("neighbour_type", "unknown")
        type_counts[nt] = type_counts.get(nt, 0) + 1

    console.print(f"    [bold green]Neighbour relations built:[/bold green]")
    console.print(f"      Total pairs: {total_pairs:,}")
    console.print(f"      Intra-site: {n_intra_site:,}")
    console.print(f"      Inter-site: {n_inter_site:,}")
    for nt, count in sorted(type_counts.items()):
        console.print(f"        {nt}: {count:,}")

    # Compute average neighbours per cell
    if all_cells:
        # Each pair contributes 1 neighbour to each cell in the pair
        avg_neighbours = total_pairs * 2 / len(all_cells)
        console.print(f"      Avg neighbours/cell: {avg_neighbours:.1f}")
