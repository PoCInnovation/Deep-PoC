# Deep-PoC

# Table of Content

[Description](#Description)\
[Installation](#Installation)\
[Quick Start](#Quick-Start)\
[Maintainers](#Maintainers)

# Description
Deep-PoC is a deepFake detection tool designed to detect deepfakes from videos or images using artificial intelligence.

The detection relies on two main weakness of deepfake : The mouth and the eyes. Deep-PoC focuses on these main parts to make its prediction.

The project is comprised of 2 parts : The Web-App, and AI.

## AI

The project's AI was created using the library called Pytorch. The AI is a imple CNN with 4 layers of convolution and 3 linear layer.
The prediction is between 0 and 1, (0 being the detction of a deepfake and 1 being the detection of a real face).

The Ai suffers from a lack of diversity in the dataset, most of the deepfake comme from https://thispersondoesnotexist.com, therefore it lacks the high capacity to detect deepfakes generated differently.

## Web-App

The web-app was created with django with the help of the dropzonejs library (https://www.dropzonejs.com/).

It is comprised of a simple drag and drop feature, to add the video of your liking to be analyzed by the AI.

# Quick Start

You'll need Python3 or hight and pip3 installed. Install the requirements with `pip3 install -r ./src/requirements.txt`

# Installation:

## Web Application

        git clone git@github.com:PoCInnovation/Deep-PoC.git
        cd Deep-PoC
        pip3 install -r requirements.txt
        cd DeepPoc
        python manage.py runserver

# Dependencies

|             Dependency           |
|:--------------------------------:|
| [opencv-python]()                |
| [opencv-contrib-python]()        |
| [numpy]()                        |
| [torch]()                        |
| [torchvision]()                  |
| [Pillow]()                       |


# Dataset

The Dataset used during this project doesn't have an easy source (For now).

To build your own dataset, oyu have a script to extract Deepfakes from https://thispersondoesnotexist.com.</br>
This only goes for deepfake, to get real images, I suggest the following dataset: https://github.com/NVlabs/ffhq-dataset.


------------


## Maintainers

 - [Victor GUyot](https://github.com/MrSIooth)
