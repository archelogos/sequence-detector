## Train the model on GCP

Use the power of Google Compute Engine using a n1-highcpu-8 instance
for one day.

Trying different configurations with Regularization techniques (L2 or/and Dropout)
as well as setting a decay factor in the Learning Rate of the Adam Optimizer.

The first step is to build a python script from the SVHN-seq notebooks that
could treat the data, create our model solution and finally train a save the
results.

SVHN-seq.py

```shell
nohup python svhn_train.py &
```

```shell
ps -U root -u root -N
```

100k steps training on GCP Compute Engine n1-highcpu-8 -> 94% Test Acc Stable
1hour and 15 minutes


200k setps training on GCP Compute Engine n1-highcpu-8 -> 96%
