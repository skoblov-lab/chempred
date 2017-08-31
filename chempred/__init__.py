import sys

if sys.version_info < (3, 5, 3):
    print("ChemPred required Python >= 3.5.3")
    sys.exit(1)
