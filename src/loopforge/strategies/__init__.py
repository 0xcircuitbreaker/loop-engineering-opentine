"""Built-in loop strategies/examples.

Drop-in step functions here are intentionally deterministic so the repo works
offline without external model credentials.
"""

from .numeric_refinement import numeric_refinement
from .text_rewrite import text_rewrite

__all__ = ["numeric_refinement", "text_rewrite"]
