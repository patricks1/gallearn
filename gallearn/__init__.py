from . import (
    cnn,
    config,
    gen_octant_images,
    preprocessing,
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
    'gen_octant_images',
    'preprocessing'
]
