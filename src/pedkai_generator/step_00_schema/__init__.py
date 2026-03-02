"""
Step 00: Schema Contracts

Defines the Parquet schema contracts for all output files.
Each schema is defined as a dictionary mapping column names to their
expected types (as PyArrow types), along with constraints and documentation.

This step produces no data — it's the contract that all downstream steps
must conform to, and that Step 10 (Validation) checks against.
"""
