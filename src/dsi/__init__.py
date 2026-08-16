"""Developmental system identification for pretraining.

Milestone A: local functional core. The modules present here are the ones
that later stages depend on for correctness, and nothing more.

Deliberately absent until their milestone: Hydra (B), Orbax (B/D), the GCP
executor and budget system (D), partial pooling (E), transfer/discovery/
curriculum (E-G), and any UI (H).
"""

SCHEMA_VERSION = 1
"""Version stamped onto every persisted table.

Changing a schema does not mutate old artifacts. Readers migrate explicitly.
"""
