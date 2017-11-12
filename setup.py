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
    version="0.5dev",
    packages=find_packages("./"),
    scripts=["chempred-example.py"],
    requires=["click==6.7",
              "fn==0.4.3",
              "frozendict==1.2",
              "h5py==2.7.0",
              "hypothesis==3.31.2",
              "joblib==0.11",
              "keras-contrib==1.2.1",
              "numpy==1.13.1",
              "pandas==0.20.3",
              "pyrsistent==0.13.0",
              "scikit-learn==0.19.0"]
              # "tensorflow==1.3.0",
              # "tensorflow-gpu==1.3.0",
              # "tensorflow-tensorboard==0.1.5"]
)
