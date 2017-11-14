# sciNER
Character-level RNN for named entity recognition in plain text.

This is a special ChemPred publication branch of the sciNER package. ChemPred
is a deep CNN-RNN architecture for chemical NER trained on the CHEMDNER corpus
(publication pending). To install the package clone ше on your local machine
and run `pip install --no-cache-dir .`. The installer will get all dependencies,
but TensorFlow - you will have to install it yourself. If you plan to use a
CUDA-enabled GPU for training/inference, you should run 
```
pip install --no-cache-dir tensorflow-gpu==1.3.0
```
otherwise run
```
pip install --no-cache-dir tensorflow==1.3.0
```
We recommend installing the package and all the dependencies in a separate
Python virtual environment (`virtualenv` or `conda`).

You can obtain the trained model from the publication at
https://1drv.ms/u/s!AlQ-UsUDf6TWgs4if4XEQ4mpz-TJog. 
We've included a basic CLI tool for the model to run inference: `chempred-example.py`, 
which will be installed with the package. The tool requires a plain text file with 
one sentence per line. We recommend using GeniaSS to break your text into separate 
sentences, because that's what the model is used to.

We've added a Jupyter notebook `chempred-training-example.ipynb` to guide you through 
the steps needed to train and validate the model.

You might be also interested in `bench.ipynb` and `bench.tgz` to see our benchmark pipeline.

Feel free to open issues here whenever problems occur.