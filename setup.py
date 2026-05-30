import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

extensions = cythonize([
    Extension("idr.inv_cdf",
              ["idr/inv_cdf.pyx", ],
              include_dirs=[numpy.get_include()],
              define_macros=define_macros),
])

setup(ext_modules=extensions)
