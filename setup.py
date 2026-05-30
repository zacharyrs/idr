import numpy
from setuptools import setup, Extension

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

try:
    from Cython.Build import cythonize
    extensions = cythonize([
        Extension("idr.inv_cdf",
                  ["idr/inv_cdf.pyx", ],
                  include_dirs=[numpy.get_include()],
                  define_macros=define_macros),
    ])
except ImportError:
    extensions = [
        Extension("idr.inv_cdf",
                  ["idr/inv_cdf.c", ],
                  include_dirs=[numpy.get_include()],
                  define_macros=define_macros),
    ]

setup(ext_modules=extensions)
