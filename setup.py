import os
import warnings
from setuptools import setup, find_packages
from distutils import sysconfig
from distutils.extension import Extension  
from Cython.Distutils import build_ext  
import numpy as np

REQUIRES = ['numpy', 'scipy', 'matplotlib', 'h5py', 'healpy',
        'pyephem', 'aipy', 'caput', 'cora', 'tlpipe']

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = REQUIRES

setup(
    name = 'fpipe',
    version = '0.1.0',

    packages = find_packages(),

    ext_modules=[Extension('fpipe.map._mapmaker', ['fpipe/map/_mapmaker.pyx'], 
                           include_dirs=[np.get_include()], 
                           depends=["setup.py",
                                    "fpipe/map/_mapmaker.pyx"]),
                 Extension('fpipe.map.cubicspline', ['fpipe/map/cubicspline.pyx'],
                           include_dirs=[np.get_include()],
                           depends=["setup.py",
                                    "fpipe/map/cubicspline.pyx"]),
                ],
    cmdclass={'build_ext': build_ext},

    install_requires = requires,
    package_data = {},

    # metadata for upload to PyPI
    author = "Yi-Chao LI",
    description = "Analysis pipeline for FAST",
)
