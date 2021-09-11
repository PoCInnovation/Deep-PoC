# Deep-PoC

# Table of Content

[Description](#Description)\
[Installation](#Installation)\
[Quick Start](#Quick-Start)\
[Maintainers](#Maintainers)

# Description 🏭
Deep-PoC is a deepFake detection tool designed to detect deepfakes from videos or images using artificial intelligence.

The detection relies on two main weakness of deepfake : The mouth and the eyes. Deep-PoC focuses on these main parts to make its prediction.

The project is comprised of 2 parts : The Web-App, and AI.

## AI

The project's AI was created using the library called Pytorch. The AI is a imple CNN with 4 layers of convolution and 3 linear layer.
The prediction is between 0 and 1, (0 being the detction of a deepfake and 1 being the detection of a real face).

The Ai suffers from a lack of diversity in the dataset, most of the deepfake comme from https://thispersondoesnotexist.com, therefore it lacks the high capacity to detect deepfakes generated differently.

![](.github/assets/eyes.png?raw=true "Real and deppfake eyes")

## Web-App

The web-app was created with django with the help of the dropzonejs library (https://www.dropzonejs.com/).

It is comprised of a simple drag and drop feature, to add the video of your liking to be analyzed by the AI.

Here is the list of supported extension:
|        Extension                        | Operational |
|:---------------------------------------:|:-----------:|
|.mp4   | :heavy_check_mark: |
| .jpeg | :heavy_check_mark: |
| .jpg  | :heavy_check_mark: |
| .png  | :x: |

![](.github/assets/frontend.png?raw=true "Real and deppfake eyes")

# Quick Start 🏁

You'll need Python3 or higher, pip3 and docker installed. Install the requirements with `pip3 install -r requirements.txt`

# Installation 🛠️

## Web Application

        git clone git@github.com:PoCInnovation/Deep-PoC.git
        cd Deep-PoC/DeepPoc
        python manage.py runserver
        docker-compose up

# Scripts

Different scripts can be used to create / update and train your own dataset and AI. These scripts are located in the: `./src/scripts/` directory.

They are to be launched from the root of the project, here is an exemple:

        python ./src/scripts/manual_test.py -h

Each script possesses a `(-h or --help)` option to view the usage of the script.

# Dataset

The Dataset used during this project doesn't have an easy source (For now).

To build your own dataset, oyu have a script to extract Deepfakes from https://thispersondoesnotexist.com.</br>
This only goes for deepfake, to get real images, I suggest the following dataset: https://github.com/NVlabs/ffhq-dataset.


------------


## Maintainers 🧑‍🤝‍🧑

 - [Victor Guyot](https://github.com/MrSIooth)
