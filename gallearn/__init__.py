import importlib

from . import (
    cnn,
    config,
    dataset_lock,
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

# gen_octant_images pulls in mockobservation_tools.galaxy_tools, which
# imports firestudio.studios.star_studio, which calls
# matplotlib.use('Agg') at import time with no way to opt out. Eagerly
# importing gen_octant_images here would silently force every
# `import gallearn` into a non-interactive matplotlib backend, even
# for callers who only want gallearn.visual_checks or
# gallearn.splitting and never touch gen_octant_images. Deferring the
# import to first access (PEP 562 module __getattr__) keeps
# `gallearn.gen_octant_images` working exactly as before for callers
# that do use it, without paying that cost for callers who don't.
def __getattr__(name):
    if name == 'gen_octant_images':
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(
        f'module {__name__!r} has no attribute {name!r}'
    )
