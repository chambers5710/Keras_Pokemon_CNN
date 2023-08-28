import os
import cv2
import pickle
import numpy as np
from tensorflow import keras
from keras.applications import MobileNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


dataset_dir = '../data/PKMN_151'
preprocessed_dir = './cache/preprocessed_data'
preprocessed_file = os.path.join(preprocessed_dir, 'preprocessed_data.npy')

data = []
labels = []

batch_size = 128
epochs = 256

os.makedirs(preprocessed_dir, exist_ok=True)

# Load pre-processed data from file if it exists
if os.path.isfile(preprocessed_file):
    print('pre-processed data found and loading...')
    data = np.load(preprocessed_file)
    labels = np.load(os.path.join(preprocessed_dir, 'labels.npy'))

# Pre-process new data
else:
    print('data loading...')
    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)               
                if image_file.lower().endswith(('.jpg', '.jpeg')):
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (224, 224))
                    image = image.astype('float32') / 255.0
                    
                    # Add the preprocessed image and its label to the lists
                    data.append(image)
                    labels.append(class_folder)

    data = np.array(data)
    labels = np.array(labels)

    # Save Pre-Processed data
    np.save(preprocessed_file, data)
    np.save(os.path.join(preprocessed_dir, 'labels.npy'), labels)

# Perform label encoding on the string labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = keras.utils.to_categorical(y_train, num_classes=149)
y_test = keras.utils.to_categorical(y_test, num_classes=149)

# Load the MobileNet model with pre-trained weights
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create the top layers for classification
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(149, activation='softmax')(x)

# Define data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True
)

# Generate augmented images during training
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

# Combine the base model and top layers
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data generator
model.fit(train_generator, steps_per_epoch=len(x_train) // batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc, "Test loss:", test_loss)

# Save model
user_input = input("Do you want to save the trained model? (y/n): ")
if user_input.lower() == 'y':
    model.save('./model_save') 
    print('Model saved.')

# Save the label encoder object
with open('./cache/label_encoder.pk1', 'wb') as f:
    pickle.dump(label_encoder, f)
