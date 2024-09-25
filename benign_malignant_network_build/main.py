from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
# Import VGG16 for transfer learning
from tensorflow.keras.applications import VGG16
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths for train and test directories
train_path = r"/Users/gabe/DS2 Project/Breast_Cancer_CNN/benign_malignant_network_build/train"
test_path = r"/Users/gabe/DS2 Project/Breast_Cancer_CNN/benign_malignant_network_build/test"

# Function to get paths for a specified image type (image_files, cropped_files, ROI_mask_files) and classification


def get_image_paths(image_type, classification, data_split='train'):
    base_path = train_path if data_split == 'train' else test_path
    return os.path.join(base_path, image_type, classification)

# Updated train_network function with changes


def train_network(DIMENSION=224, image_type="image_files"):
    # Update paths based on selected image type
    train_dir = os.path.join(train_path, image_type)
    test_dir = os.path.join(test_path, image_type)

    # Image augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30
    )

    # Load training data directly using ImageDataGenerator
    train_dataset = datagen.flow_from_directory(
        train_dir,
        target_size=(DIMENSION, DIMENSION),
        batch_size=64,  # Increase batch size to 64
        class_mode='categorical'  # Use 'categorical' since we have more than 2 classes
    )

    # Load test data directly using ImageDataGenerator
    test_dataset = datagen.flow_from_directory(
        test_dir,
        target_size=(DIMENSION, DIMENSION),
        batch_size=64,  # Increase batch size to 64
        class_mode='categorical'
    )

    # Use transfer learning with VGG16
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(DIMENSION, DIMENSION, 3))
    base_model.trainable = False  # Freeze the layers of the pre-trained model

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    # Reduced units for faster training
    model.add(Dense(64, activation='relu'))
    # Use 'softmax' for multi-class classification
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy", metrics=["accuracy"])

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=5)

    # Model training
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=10,  # Reduced number of epochs to 10 for faster training
        callbacks=[early_stop]
    )

    # Model evaluation
    y_pred = model.predict(test_dataset)
    y_pred_r = np.argmax(y_pred, axis=1)
    y_test = test_dataset.classes  # Get true labels

    # Print accuracy and evaluation metrics
    acc_score = accuracy_score(y_test, y_pred_r)
    print("Accuracy Score: ", acc_score)
    print("Classification Report: ")
    print(classification_report(y_test, y_pred_r))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred_r))
    print("==================================================\n")

    return acc_score


# Run the training using image files
train_network()
