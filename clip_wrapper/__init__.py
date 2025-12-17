"""
CLIP Image Encoder Module for TomBERT
Provides CLIP-based image encoding as a replacement for ResNet-152
Expected improvement: 5-10% accuracy boost
"""

from .clip_utils import CLIPImageEncoder, CLIP_AVAILABLE

__all__ = ['CLIPImageEncoder', 'CLIP_AVAILABLE']

