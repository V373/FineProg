# -*- coding: utf-8 -*-
"""TCC PyTorch: Temporal Cycle-Consistency Learning in PyTorch"""

__version__ = '0.1.0'
__author__ = 'TCC Team'

try:
    from .config import CONFIG
    __all__ = ['CONFIG']
except ImportError:
    pass
