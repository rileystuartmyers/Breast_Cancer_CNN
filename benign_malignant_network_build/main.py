from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shutil

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Paths for train and test directories
train_path = r"/Users/gabe/DS2 Project/Breast_Cancer_CNN/benign_malignant_network_build/train"
test_path = r"/Users/gabe/DS2 Project/Breast_Cancer_CNN/benign_malignant_network_build/test"

# Function to get paths for a specified image type (image_files, cropped_files, ROI_mask_files) and classification


def get_image_paths(image_type, classification, data_split='train'):
    base_path = train_path if data_split == 'train' else test_path
    return os.path.join(base_path, image_type, classification)


# Image paths for training data
benign_image_files_train = get_image_paths("image_files", "BENIGN", "train")
benign_without_callback_image_files_train = get_image_paths(
    "image_files", "BENIGN_WITHOUT_CALLBACK", "train")
malignant_image_files_train = get_image_paths(
    "image_files", "MALIGNANT", "train")

# Image paths for testing data
benign_image_files_test = get_image_paths("image_files", "BENIGN", "test")
benign_without_callback_image_files_test = get_image_paths(
    "image_files", "BENIGN_WITHOUT_CALLBACK", "test")
malignant_image_files_test = get_image_paths(
    "image_files", "MALIGNANT", "test")


def image_paths(folder):
    image_paths = []
    for filename in os.listdir(folder):
        image_paths.append(filename)
    return image_paths


def load_images(folder, dimension):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resized_img = cv2.resize(img, (dimension, dimension))
            images.append(resized_img)
    return images


def single_image_formatting(directory, dimension):
    images = []
    img = cv2.imread(directory)
    resized_img = cv2.resize(img, (dimension, dimension))
    images.append(resized_img)
    images = np.array(images)
    images = images / 255.0
    return images


def train_network(DIMENSION, image_type="image_files"):
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
        batch_size=32,
        class_mode='binary'  # Use 'categorical' if you have more than 2 classes
    )

    # Load test data directly using ImageDataGenerator
    test_dataset = datagen.flow_from_directory(
        test_dir,
        target_size=(DIMENSION, DIMENSION),
        batch_size=32,
        class_mode='binary'
    )

    # Define CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu",
              input_shape=(DIMENSION, DIMENSION, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # Use 'softmax' for multi-class classification
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=5)

    # Model training
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=30,
        callbacks=[early_stop]
    )

    # Model evaluation
    y_pred = model.predict(test_dataset)
    y_pred_r = np.argmax(y_pred, axis=1)
    y_test = test_dataset.classes  # Get true labels

    # Print accuracy and evaluation metrics
    acc_score = accuracy_score(y_pred_r, y_test)
    print("Accuracy Score: ", acc_score)
    print("Classification Report: ")
    print(classification_report(y_test, y_pred_r))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred_r))
    print("==================================================\n")

    return acc_score


# Run the training using image files
train_network(700, image_type="image_files")
