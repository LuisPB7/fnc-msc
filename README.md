# Fake News Challenge - M.Sc. Thesis
Deep Learning model to tackle the Fake News Challenge (http://www.fakenewschallenge.org/)

-> Python 2 and Python 3

-> Keras 1.2.2

-> **fnc_model.py**: model definition and training

-> **my_layers2.py**: custom Keras layers and callback

-> **generate-everything.py**: generate some features and useful variables into files variables.pkl and features.pkl

-> **generate-similarities.py**: generate similarity features into similarity.pkl

-> **concat_snli_pooling.h5**: https://mega.nz/#!hgIwTZrT!sUsk5Ryu9elax_B2-H7eFwE41tsy2f36p5wg-4jyrio

-> **snli-pooling.h5**: https://mega.nz/#!olZylILQ!r3GKndoQh3_JreUIsxjSq3zDtwn4nTRq6bTXhYhXVR8

-> **features.pkl**: https://mega.nz/#!otoX0bCJ!6e4RVjmNgK5Gopkmueuv5YV0PxAUyN6XtiBugb8cxP4

-> **similarity.pkl**: https://mega.nz/#!Qs5kAYDR!4JbHavQB8ILmO6r5Z6ZosAAJJT-YXH6XIYJy3e0XFRQ

-> **variables.pkl**: https://mega.nz/#!9h5G0T6Q!cqW21okeT4kILtVl5_9pDHA-vf02_qks7Hls80Z8sKY

-> **test.body.word2vec.pkl**: https://mega.nz/#!I4ZWAZYY!sMYOlVcbnZkR580CifFknYmbI1CHTdyEYBnarcSGBEk

-> **test.headline.word2vec.pkl**: https://mega.nz/#!BtpwSQYC!A2cvzTsH_8S6LLjNC2cfB8wwaiFAwzPNZlR-9Qnn3UI

-> **train.body.word2vec.pkl**: https://mega.nz/#!Qlw1AQbC!XrBsiJgBrkJ04wIH-kBBYM8NWPDNnLfK1DAYSogs5Cg

-> **train.headline.word2vec.pkl**: https://mega.nz/#!B9ZjQbRS!TxgB0fTv3hCPKCS9wOF574STGf7naA2DBx9t2AK_BFs

**INSTRUCTIONS**

1 - Download the SNLI and MultiNLI datasets from the two corresponding links:

https://nlp.stanford.edu/projects/snli/

http://www.nyu.edu/projects/bowman/multinli/

2 - Download all the files in this repository, including ```test.body.word2vec.pkl```, ```test.headline.word2vec.pkl```,```train.body.word2vec.pkl``` and ```train.headline.word2vec.pkl```, which were too big to upload directly.

2 - Run ```python3 generate-everything.py``` to generate the files ```features.pkl``` and ```variables.pkl```, or simply download them from the above links.

3 - Run ```python3 generate-similarities.py``` to generate the files ```similarity.pkl```, or simply download it from the above links.

4 - Run ```python nli-model.py``` to train and test the SNLI+MultiNLI model, hence generating the files ```snli-pooling.h5``` and ```concat_snli_pooling.h5```. Once again, you can simply download them directly.

5 - Run ```python fnc-model.py``` to train and test the FNC model (code already includes SNLI+MultiNLI weight loading)
