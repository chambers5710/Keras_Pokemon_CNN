import cv2
import pickle
import numpy as np
from tensorflow import keras

running = True

# Load the saved model and label encoder
print("Loading...")
model = keras.models.load_model('model_save')

with open('./cache/label_encoder.pk1', 'rb') as f:
    label_encoder = pickle.load(f)

# Load and preprocess the input image
while running == True:

    # Example images have been included. Enter the following format: "./test_input/test_photo-1.jpg"
    user_input = input('Please provide a filepath to your image. ')
    if user_input == "exit":
            running == False
            exit()

    input_image = cv2.imread(user_input, cv2.IMREAD_COLOR)
    input_image = cv2.resize(input_image, (224, 224))
    input_image = input_image.astype('float32') / 255.0

    #Reshape and expand the dimensions of the preprocessed image to match the model's input shape
    input_image = np.expand_dims(input_image, axis=0)

    # Pass preprocessed image to the trained model
    predictions = model.predict(input_image)
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

    print("Prediction:", predicted_label)

