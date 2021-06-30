# Sign-to-Speech
---
## How I got the idea
---

Inability to speak is considered to be a true disability. The [World Health Organization’s(WHO)](https://www.who.int/data/gho) survey states that above 6% of the world’s population is suffering from hearing impairment. In March 2018, The number of people with this disability is 466 million, and it is expected to be 900 million by 2050. Although the above data is pretty old but looking at the numbers we can say that it is a matter of Importance 7 million is not a small number. People with this disability use different modes to communicate with others, there are n number of methods available for their communication. Text messaging, writing, using visual media and finger spelling are a few methods used to establish communication between normal and hearing and speech impaired people. However one common method of communication is sign language. 

Sign languages are a visual representation of thoughts through hand gestures, facial expressions and body movements. People use sign language gestures as a means of non-verbal communication to express their thoughts and emotions. Sign language allows people to communicate with human body language each word has a set of human actions representing a particular expression. Sign languages also have several variants, such as [American Sign Language(ASL)](https://www.nidcd.nih.gov/health/american-sign-language) , [Argentinian Sign Language(LSA)](https://argentinesignlanguage.bible/), [British Sign Language(BSL)](https://www.signbsl.com/) and [Indian Sign Language (ISL)](https://indiansignlanguage.org/). Moreover in India, there is no universal sign language. Though there exist many sign languages normal people do not know about sign language. They find it extremely difficult to understand, hence trained sign language interpreters are needed during medical and legal appointments, educational and training sessions. Over the past few years, there has been an increasing demand for interpreting services.

Interpretation of sign language can be done in two ways, either glove based recognition or vision based recognition. In glove based technique a network of sensors is used to capture the movements of the fingers. Facial expressions cannot be recognized in this method and also, wearing a glove is always uncomfortable for the users. Few users might not be able to afford such gloves. This project uses a vision based recognition method. The vision based recognition can be Static recognition or Dynamic recognition. 

In static recognition the input may be an image of hand pose. It provides only 2D representation of the gesture, and this can be used to recognize only alphabets and numbers. For recognition of continuous sign language, the dynamic gesture recognition system is used. Here videos are given as inputs instead of images. The recognition is achieved using [Deep learning](https://www.ibm.com/cloud/learn/deep-learning) and Computer vision, Hence if a person who has a disability to speak can stand in front of the system and the system converts the human gestures as speech and plays it loud so that the person could actually communicate to a mass crowd gathering.

As we explained prior, A sign is a movement of one or both hands, accompanied with facial expression, which corresponds to a specific meaning. Even thought facial expressions add important information to the emotional aspect of the sign, but in this project we have excluded this aspect from the area of interest, as its analysis makes the already difficult problem more complicated.

Although one might argue about the requirement of sign language in this modern world of smartphones, where everyone can text message to express their thoughts and ideas. Anyone can use applications like whatsapp, telegram or even mails to message other people. At the same time it is evident to see that not all people across the world are literate enough to use those applications. Also if the speech-impaired people write what they intend to speak, there might be a possibility that the other person cannot understand what he has written / might misinterpret what the disabled has written.
The solution to these problems is an efficient sign language interpreter which can recognize the gestures shown by  deaf and speech-impaired people and convert that into suitable forms like text / audio.

## Literature survey
---
- [Real-Time Recognition of Indian Sign Language](https://ieeexplore.ieee.org/document/8862125)
- [Continuous dynamic Indian Sign Language gesture recognition with invariant backgrounds](https://ieeexplore.ieee.org/document/7275945)
- [Machine learning Techniques for Indian Sign Language Recognition](https://ieeexplore.ieee.org/document/8454988)
- [Real time Conversion of Sign Language using Deep Learning for Programming Basics](https://ieeexplore.ieee.org/document/9087272)

Above are few of the papers our team referred to do this project. We have also referred several other Conference papers.
**Also you can refer this article [Sign language recognition using Python and Opencv](https://data-flair.training/blogs/sign-language-recognition-python-ml-opencv/)**

## Description
---
The recognition of the sign language is done using a CNN(Convolutional Neural Network) model which is trained on a dataset containing 35 classes among which 26 are for alphabets and remaining are for numbers. The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets). Tensorflow has been utilized to train the model.
Sequential model with three *Convo2D* layers, three *MaxPool2D* layers and three dense layers with *relu* activation funcion has been used. The output Dense layer has softmax activation function with 35 neurons.
[Adam](https://keras.io/api/optimizers/adam/) optimizer with [categorical crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) loss function is used. The model is then trained for 10 epochs.

***Dataset can be downloaded from [here](https://www.kaggle.com/vaishnaviasonawane/indian-sign-language-dataset)***.

## Strucure of repo
---
The repository contains the following structure.
- README.md - This is a markdown file which contains details about the project.
- My final model

<!--[Webcam capture](/images/Webcamcapture.png)
![Hand image](/images/fg.png) -->

## Future Work
---
At this point of time the model is able to recognize gestures from static images. The prediction is above 95% accurate for static inputs(images). We are working to produce a system which can recognize gestures shown by user in real time.
Image is captured from webcam and only the hand reagion of that image is given as input for the CNN model. The model must recognize the sign.
We are still working to capture the hand region from live webcam.