import sys
from distutils.core import setup
from setuptools import find_packages

if sys.version_info < (3, 5, 3):
    print("ChemPred required Python >= 3.5.3")
    sys.exit(1)

# from Cython.Build import cythonize
#
# os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="chempred",
    version="0.1dev",
    packages=find_packages("./"),
    scripts=["chem-pred"],
    requires=["numpy",
              "h5py",
              "fn",
              "enforce",
              "pyrsistent",
              "keras"]
)
