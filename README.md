# Detecting Fake News Using Deep Neural Networks

This work was developed by Luís Borges in the context of his M.Sc. thesis at Instituto Superior Técnico of the University of Lisbon. His work focused on the creation of a model to tackle the stance detection task proposed by the Fake News Challenge -- given a headline and the body of an article, to determine the stance of the headline relative to the body into one of {*agrees*, *disagrees*, *discusses*, *unrelated*}.

The source code in this repository is associated with Luís Borges' M.Sc. thesis and with the work described in the following publication:

```
@article{Borges2019,
   author = {Borges, Luís and Martins, Bruno and Calado, Pável},
   journal = {ACM Journal of Data Information and Quality},
   title = {Combining Similarity Features and Deep Representation Learning for Stance Detection in the Context of Checking Fake News},
   year = {2019}
}
```

**INSTRUCTIONS**

1 - Download the SNLI and MultiNLI datasets from the two corresponding links:

https://nlp.stanford.edu/projects/snli/

http://www.nyu.edu/projects/bowman/multinli/

2 - Download and extract a set of features used as input to the model:
https://mega.nz/#!08IyyKrT!rfsor4KQSPX0OCBAMhIzqhfOLkRZQBAN5BqqiIJXdrA 

3 - Run ```python3 generate.py``` to generate necessary files for model training and testing. You can also download them directly:
https://mega.nz/#!tlAG3abL!hoPZhr7X5M1ifuPaXSAaNExuAz8DdEHHdV3EC5PSojE

4 - Run ```python3 nli-model.py``` to train and test the SNLI+MultiNLI model.

5 - Run ```python3 fnc-model.py``` to train and test the FNC model (code already includes SNLI+MultiNLI weight loading).

The source code was run with Python 3 and Keras 2.
