import sys
from distutils.core import setup
from setuptools import find_packages

# TODO add loggers and warnings
# TODO lazy module improting (https://github.com/bwesterb/py-demandimport)

if sys.version_info < (3, 5, 2):
    print("ChemPred required Python >= 3.5.2")
    sys.exit(1)

# from Cython.Build import cythonize
#
# os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="sciner",
    version="0.1dev",
    packages=find_packages("./"),
    scripts=["chem-pred"],
    requires=["numpy",
              "h5py",
              "fn",
              "pyrsistent",
              "keras",
              "intervaltree",
              "scikit-learn", 'pandas']
)
