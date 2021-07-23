
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
The recognition of the sign language is done using a CNN(Convolutional Neural Network) model which is trained on a dataset containing 35 classes among which 26 are for alphabets and remaining are for numbers. A custom Dataset was created for training purposes. Tensorflow has been utilized to train the model.
<br>Below images show the foreground extraction from frames (which was used for dataset creation).

<img src="assets/Hand.png" alt="Hand" width=150 height=150>
<img src="assets/Binary hand.png" alt="binary image " width=150 height=150>

Sequential model with three *Convo2D* layers, three *MaxPool2D* layers and three dense layers with *relu* activation funcion has been used. The output Dense layer has softmax activation function with 35 neurons.
[Adam](https://keras.io/api/optimizers/adam/) optimizer with [categorical crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) loss function is used. The model is then trained for 10 epochs. The model was later trained for 25 epochs and got more accuracy.


<h2>Strucure of repo</h2>
<hr>
The repository contains the following structure.
- README.md - This is a markdown file which contains details about the project.
- Main scripts - This folder contains the main script files which can be used to either generate the model or for prediction with the model.
- Mytrials - This folder contains the files which which were used while training the model and also for testing the codes.

<!--[Webcam capture](/images/Webcamcapture.png)
![Hand image](/images/fg.png) -->

<h2>Future Work</h2>
<hr>
The model is currently capable to recognize the gestures of numbers, alphabets and few words of Indian Sign langauge. The accuracy is about 89%, Image is captured from webcam and only the hand reagion of that image is given as input for the CNN model. The model recognizes the gesture.
However the system works fine for plain background, the accuracy of the system drops drasticaaly for variant backgrouns. The future works may focus on making the system background invariant.

<h2>References</h2> 
<hr>
- [Real-Time Recognition of Indian Sign Language](https://ieeexplore.ieee.org/document/8862125)
- [Continuous dynamic Indian Sign Language gesture recognition with invariant backgrounds](https://ieeexplore.ieee.org/document/7275945)
- [Machine learning Techniques for Indian Sign Language Recognition](https://ieeexplore.ieee.org/document/8454988)
- [Real time Conversion of Sign Language using Deep Learning for Programming Basics](https://ieeexplore.ieee.org/document/9087272)

Above are few of the papers our team referred to do this project. We have also referred several other Conference papers.
**Also you can refer this article [Sign language recognition using Python and Opencv](https://data-flair.training/blogs/sign-language-recognition-python-ml-opencv/)**

