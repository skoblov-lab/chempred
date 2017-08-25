from distutils.core import setup
from setuptools import find_packages

# from Cython.Build import cythonize
#
# os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="chempred",
    version="0.1dev",
    packages=find_packages("./"),
    requires=["numpy",
              "h5py",
              "fn",
              "enforce",
              "pyrsistent",
              "keras"]
)
