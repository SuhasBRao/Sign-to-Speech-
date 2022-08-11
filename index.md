
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
<link rel="stylesheet" href="assets/style.css" >

<a href="https://suhasbrao.github.io/" >
    <button class="btn"><i class="material-icons">home</i></button></a>
  <a href="https://suhasbrao.github.io/FaceDetection/" >
    <button class="btn"><i class="material-icons">arrow_back_ios</i></button></a>
  <a href="https://suhasbrao.github.io/Snake-game/" >
    <button class="btn"><i class="material-icons">arrow_forward_ios</i></button></a>
<hr class="hr1" />

<h2>Description</h2>
<hr>
The recognition of the sign language is done using a <b>CNN(Convolutional Neural Network)</b> model which is trained on a dataset containing <b>41 classes
among which 26 classes are for alphabets, 6 classes for words and remaining are for numbers</b>. 
A custom dataset has been generated and the code for the same is available in <i>CNN model.ipynb</i> file.

A sample pic of the dataset generated is shown below.

<p align ="center">
<img src="./assets/Binary%20hand.png" alt="drawing" width="150" height="150" /> 
</p>


A pic of prediction of local image

<p align ="center">
<img src="./assets/classify.png" alt="drawing" width="250" height="250" /> 
</p>

<h2>CNN model specification</h2>
<hr>
Sequential model with 

- Three *Convo2D* layers
- Three *MaxPool2D* layers
- Three dense layers with *relu* activation funcion has been used. 
 
 The output Dense layer has softmax activation function with 35 neurons.

[Adam](https://keras.io/api/optimizers/adam/) optimizer with [categorical crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) loss function is used. The model is then trained for 10 epochs.


<h2>Strucure of repo</h2>
<hr>
The repository contains the following structure.

- README.md - This is a markdown file which contains details about the project.
- Main scripts - This folder contains the main script files which can be used to either generate the model or for prediction with the model.
- Mytrials - This folder contains the files which which were used while training the model and also for testing the codes.

<!--[Webcam capture](/images/Webcamcapture.png)
![Hand image](/images/fg.png) -->

<h2>Contribution</h2>
<hr>
Do you have any suggestions on improving this project?

Open an issue if you have any suggestions. 

Here are few things You can work on
- Improving the model's accuracy.
- Try with other algorithms like SVM, K nearest neighbor etc.
- To build a web version of the project(For this you need to consider the application.py file )

Feel free to modify the project and open a Pull request.

**Also you can refer this article [Sign language recognition using Python and Opencv](https://data-flair.training/blogs/sign-language-recognition-python-ml-opencv/)**

