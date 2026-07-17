from . import (
    cnn,
    config,
    dataset_lock,
    gen_octant_images,
    preprocessing,
    splitting,
    train,
    visual_checks,
)

try:
    from ._version import __version__
except ImportError:
    __version__ = 'unknown'

# List the modules and objects you want to make available when using wildcard 
# imports
__all__ = [
    'cnn',
    'config',
    'dataset_lock',
    'gen_octant_images',
    'preprocessing',
    'splitting',
    'train',
    'visual_checks',
]
