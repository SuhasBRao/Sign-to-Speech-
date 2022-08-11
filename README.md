# Sign-to-Speech
---
## What is Sign-to-Speech ?
---
As the name suggests, Sign-to-Speech is a project that converts Signs/Gestures to Speech. 

#### **Why would we need that ?**

Ever wondered how speech impaired people communicate with each other. You are right, They communicate using Gestures or Sign Language. 

If they try to communicate with us would we be able understand what they are telling us. Not unless we know a Sign Language.

Sign-to-Speech tries to overcome that gap.

 Where it allows the users to show gestures to the camera and converts that to speech. 
 Its basically a ***Gesture Recognition*** app.

## Description
---
The recognition of the sign language is done using a **CNN(Convolutional Neural Network)** model which is trained on a dataset containing ***41 classes
among which 26 classes are for alphabets, 6 classes for words and remaining are for numbers***. 
A custom dataset has been generated and the code for the same is available in *CNN model.ipynb* file.

A sample pic of the dataset generated is shown below.

<p align ="center">
<img src="./assets/Binary%20hand.png" alt="drawing" width="150" height="150" /> 
</p>


A pic of prediction of local image

<p align ="center">
<img src="./assets/classify.png" alt="drawing" width="250" height="250" /> 
</p>

## CNN model specification
---
Sequential model with 
- Three *Convo2D* layers
- Three *MaxPool2D* layers
- Three dense layers with *relu* activation funcion has been used. 
 
 The output Dense layer has softmax activation function with 35 neurons.

[Adam](https://keras.io/api/optimizers/adam/) optimizer with [categorical crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) loss function is used. The model is then trained for 10 epochs.


## Strucure of repo
---
The repository contains the following structure.
- README.md - This is a markdown file which contains details about the project.
- Main scripts - This folder contains the main script files which can be used to either generate the model or for prediction with the model.
- Mytrials - This folder contains the files which which were used while training the model and also for testing the codes.
- other files - Files lke _config.yml , index.md etc are used by github-pages


## Contribution
---
Do you have any suggestions on improving this project?

Open an issue if you have any suggestions. 

Here are few things You can work on
- Improving the model's accuracy.
- Try with other algorithms like SVM, K nearest neighbor etc.
- To build a web version of the project(For this you need to consider the application.py file )

Feel free to modify the project and open a Pull request.
