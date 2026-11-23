"""
Adaptive Enhancement Module.

Provides scene-characteristic driven Retinex illumination and reflectance processing.
"""

from .hybrid_retinex import hybrid_adaptive_retinex, analyze_scene_characteristics, fast_guided_filter

__all__ = [
    "hybrid_adaptive_retinex",
    "analyze_scene_characteristics",
    "fast_guided_filter"
]
