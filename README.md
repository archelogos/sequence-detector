# Digit Sequence Detector

## Getting Started

The main goal of this project is, thanks to Deep Learning techniques, to be able **to detect and identify sequences of digits in a random picture**. Specifically in this project, the images correspond to houses and the number sequences correspond to their house numbers.

The data used to train the model is part of the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, which is a real-world image dataset for developing machine learning and object recognition problems and it is obtained from house numbers in Google Street View images.

It is available a full document which explains the followed process in this project. [Document](https://github.com/archelogos/sequence-detector/blob/master/project/docs/Sequence_Detector-MLDN_Capstone_Project.pdf)

There is also a public Web Application where can be seen the simplified goal of this project. [https://sequence-detector.herokuapp.com](https://sequence-detector.herokuapp.com)

## Install

This project requires **Python 2.7**, **[Conda](http://conda.pydata.org/docs/intro.html)** and the following Python libraries:

- [TensorFlow](http://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)

The most relevant parts of this project are developed in [iPython Notebooks](http://ipython.org/notebook.html)

## Code

The structure of the project is divided in three different parts:

 - **notebooks**: this part contains all the developed notebooks along the process building from a **Logistic Regression** to a **Multilayer ConvNet**.
 - **train-scripts**: a series of python scripts that allow to train the final model on [Google Cloud Platform](https://cloud.google.com/).
 - **app**: a web app coded in Python and "vanilla" Javascript, and all the necessary code to deploy it on [Heroku](https://www.heroku.com/).

## Run

How to train the model on GCP:

 - Create VM instances on Google Compute Engine
 - Access to the instances from shell
 - Run a background training process: `nohup python svhn_train.py &`

How to run and deploy the Web App on Heroku:

 - Create a Virtual Environment: `virtualenv env`
 - `source env/bin/activate`
 - Install the dependencies: `pip install -r requirements.txt`
 - Push the repo to Heroku remote: `git push heroku master`

## Data

The used datasets in this project are:

 - [MNIST data](http://yann.lecun.com/exdb/mnist/)
 - [SVHN data](http://ufldl.stanford.edu/housenumbers/)

## Others

About:

 - [Linkedin](https://es.linkedin.com/in/sgordillo)
 - [Twitter](https://twitter.com/Sergio_Gordillo)

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

		https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
