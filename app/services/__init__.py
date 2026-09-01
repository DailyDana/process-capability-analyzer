"""Services layer: orchestration across the domain subpackages.

This layer is allowed to depend on multiple domain subpackages (spc, capability,
statistics) and wire them together. Keeping the coupling here preserves the
one-directional dependency rule for the pure domain modules.
"""
