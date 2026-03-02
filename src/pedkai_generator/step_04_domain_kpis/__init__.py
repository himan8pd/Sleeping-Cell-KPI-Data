"""
Step 04: Multi-Domain KPI Generation.

Generates hourly KPIs for all non-radio domains:
- Transport (MPLS/IP links, microwave, fibre)
- Fixed Broadband (OLT, PON, ONT)
- Enterprise Circuits (L3VPN, E-Line)
- Core Network (EPC, 5GC, IMS elements)
- Power/Environment (per-site power, cooling, battery)

Each domain's KPIs are causally consistent with the topology
generated in Step 02 and with radio KPIs from Step 03.
Cross-domain correlations are maintained (e.g., a congested
PE router causes elevated latency on cells it backhauls).
"""
