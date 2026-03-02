"""
Step 02: Full Network Topology Generation.

Builds the complete converged-operator topology graph across all 6 domains:
- Mobile RAN (~170K entities)
- Fixed Broadband Access (~120K entities)
- Transport (~50K entities)
- Core Network (~200 entities)
- Logical/Service (~10K entities)
- Power/Environment (~40K entities)
- Customers (~1.1M entities)

Total: ~1.49M entities, ~2.21M relationships.

Output:
- ground_truth_entities.parquet
- ground_truth_relationships.parquet
- neighbour_relations.parquet
"""
