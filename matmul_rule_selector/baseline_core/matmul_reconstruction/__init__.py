"""Versioned MatMulV3 reconstruction components.

The official selector is intentionally not imported as a package side effect.
Tools that explicitly need the reference implementation must import
``matmul_reconstruction.api``; the independent generator imports only ABI,
initializer and finite incremental-loop components.
"""

__all__ = []
