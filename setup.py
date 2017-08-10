from setuptools import find_packages
from distutils.core import setup
import os

from Cython.Build import cythonize

os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="chempred",
    ext_modules=cythonize(["chempred/chemdner.pyx"]),
    packages=find_packages("./"),
    requires=["numpy",
              "cython",
              "fn"]
)
