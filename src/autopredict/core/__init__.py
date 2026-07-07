"""Domain core: framework-agnostic entities, enums, ports and errors.

Nothing in this package may import from adapter layers (ingestion, api,
training, ...). Dependencies point *inward* only — the dependency rule of
Clean Architecture.
"""
