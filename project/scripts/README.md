## Train the model on GCP

Use the power of Google Compute Engine using a n1-highcpu-32 instance
for one day.

Trying different configurations with Regularization techniques (L2 or/and Dropout)
as well as setting a decay factor in the Learning Rate of the Adam Optimizer.

The first step is to build a python script from the SVHN-seq notebooks that
could treat the data, create our model solution and finally train a save the
results.

SVHN-seq.py
