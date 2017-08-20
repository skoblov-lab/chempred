from distutils.core import setup
from setuptools import find_packages

# from Cython.Build import cythonize
#
# os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="chempred",
    # ext_modules=cythonize(["chempred/chemdner.pyx"]),
    packages=find_packages("./"),
    requires=["numpy",
              "h5py",
              # "cython",
              "fn",
              "enforce",
              "pyrsistent",
              "keras"]
)
