# Convolutional Neural-Network Poke-Dex Trained on Gen-1 Pokemon

Here we have a CNN using Keras' MobileNet trained to recognize front-facing images of Pokemon. It has been trained on an image dataset of the first generation, including around 10,000 images. 

## Installation

To test the neural network on a Pokemon image of your own, the following Python libraries are needed:
### OpenCV-Python, Pickle, & NumPy

To train the neural network on your own data, the following Python libraries are needed:
### NumPy, OpenCV-Python, TensorFlow, Keras, & scikit-learn

```bash
pip install opencv-python
pip install numpy
pip install tensorflow
pip install keras
pip install scikit-learn
```

## Usage

### test.py
While the program runs, the user may input a filepath leading to an image of a Pokemon. The script will load it and pass it through the neural network, returning its best guess of which of the first 151 Pokemon is pictured.

ensure the following libraries are properly imported:
```python
import cv2
import pickle
import numpy as np
from tensorflow import keras
```
then press "Run". You will be prompted to input a filepath leading to the image you wish to be interpreted. The program will then return it's best guess after processing. 

### train.py
Use this script to train the neural network on your own dataset. Simply update ```dataset_dir``` and adjust the parameters, including ```batch_size``` and ```epochs``` if needed.

## Commands

Use 'exit' to end the program runtime.

## License

[MIT](https://choosealicense.com/licenses/mit/)