import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import shutil

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,BatchNormalization    # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator                          # type: ignore
from tensorflow.keras.models import Sequential                                               # type: ignore
from tensorflow.keras.callbacks import EarlyStopping                                         # type: ignore

source_folder_parent = r"breast_cancimg\jpeg"

#dicom_info = pd.read_csv(r"breast_cancimg\csv\dicom_info.csv")
#mass_case_desc_test_set = pd.read_csv(r"breast_cancimg\csv\mass_case_description_test_set.csv")
#mass_case_desc_train_set = pd.read_csv(r"breast_cancimg\csv\mass_case_description_train_set.csv")
#meta = pd.read_csv(r"breast_cancimg\csv\meta.csv")
#calc_case_desc_test_set = pd.read_csv(r"breast_cancimg\csv\calc_case_description_test_set.csv")
#calc_case_desc_train_set = pd.read_csv(r"breast_cancimg\csv\calc_case_description_train_set.csv")

train_path = r"breast_cancer_CNN\train\image_files"
test_path = r"breast_cancer_CNN\test\image_files"

benign_train_path = r"breast_cancer_CNN\train\image_files\BENIGN"
#benign_without_callback_train_path = r"breast_cancer_CNN\train\image_files\BENIGN_WITHOUT_CALLBACK"
malignant_train_path = r"breast_cancer_CNN\train\image_files\MALIGNANT"

benign_test_path = r"breast_cancer_CNN\test\image_files\BENIGN"
#benign_without_callback_test_path = r"breast_cancer_CNN\test\image_files\BENIGN_WITHOUT_CALLBACK"
malignant_test_path = r"breast_cancer_CNN\test\image_files\MALIGNANT"


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

            #greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(img, (dimension, dimension))
            images.append(resized_img)

    return images

def single_image_formatting(directory, dimension):

    images = []
    
    img = cv2.imread(directory)
    #greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img, (dimension, dimension))
    
    images.append(resized_img)

    images = np.array(images)

    images = images / 255.0
    
    return images

def final_prediction(predicted_value):

    output_statement = ""
    
    val = np.round(predicted_value)

    if (val == 0):

        output_statement = "happy"

    else:

        output_statement = "sad"

    return output_statement

def copy_tabled_images_to_path(tableset, source_folder_parent, train_or_test, image_type):

    #source_folder_parent is stand-in for folder location of 'jpeg'

    #image_type refers to one of three image types among images in the 'jpeg' folder: image, cropped, and ROI_mask

    image_csv_string = ""

    if (image_type == "image"):
    
        image_csv_string = "image file path"

    elif (image_type == "cropped"):
    
        image_csv_string = "cropped image file path"

    else:
    
        image_csv_string = "ROI mask file path"

    
    for case in range(len(tableset)):

        #patient_id = tableset['patient_id'][case]
        path_segments = tableset[image_csv_string][case]
        pathology = tableset['pathology'][case]

        first_index = path_segments.find("/")
        second_index = path_segments.find("/", first_index + 1)
        third_index = path_segments.find("/", second_index + 1)

        direct_folder_path = path_segments[second_index + 1:third_index]
        
        source_folder = source_folder_parent + "\\" + direct_folder_path

        print(source_folder) 

        if ((os.path.exists(source_folder)) == False):

            print(direct_folder_path + " does not exist.")
            continue

        source_dir = os.listdir(source_folder)

        if ((image_type != "image") and (image_type != "cropped")):

            for image in source_dir:

                if (image[0] == "2"):

                    image_path = source_folder + "\\" + image

                    shutil.copy(image_path, "breast_cancer_CNN" + "\\" + train_or_test + "\\" + image_type + "_files" + "\\" + pathology + "\\" + image)

        else:

            for image in source_dir:

                if (image[0] == "1"):

                    image_path = source_folder + "\\" + image

                    shutil.copy(image_path, "breast_cancer_CNN" + "\\" + train_or_test + "\\" + image_type + "_files" + "\\" + pathology + "\\" + image)


def train_network(DIMENSION):

    benign_train = load_images(benign_train_path, DIMENSION)
    #benign_without_callback_train = load_images(benign_without_callback_train_path, DIMENSION)
    malignant_train = load_images(malignant_train_path, DIMENSION)

    benign_test = load_images(benign_test_path, DIMENSION)
    #benign_without_callback_test = load_images(benign_without_callback_test_path, DIMENSION)
    malignant_test = load_images(malignant_test_path, DIMENSION)

    X_train = benign_train + malignant_train #+ benign_without_callback_train
    X_test = benign_test + malignant_test #+ benign_without_callback_test

    X_train = np.array(X_train)
    X_train = X_train / 255.0

    X_test = np.array(X_test)
    X_test = X_test / 255.0

    datagen = ImageDataGenerator(

        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 30

    )

    train_dataset = datagen.flow_from_directory(train_path, class_mode = "binary")
    test_dataset = datagen.flow_from_directory(test_path, class_mode = "binary")

    y_train = train_dataset.classes
    y_test = test_dataset.classes

    model=Sequential()

    model.add(Conv2D(32,(3,3),activation="relu",input_shape=(DIMENSION, DIMENSION, 3)))
    
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64,(3,3),activation="relu"))
    
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(2, activation = "sigmoid"))

    model.summary()

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=5)

    history=model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=30, callbacks = [early_stop], shuffle=True)

    y_pred = model.predict(X_test)

    y_pred_r = np.argmax(y_pred, axis = 1) 

    output_list = list(zip(y_pred_r, y_test))
    for x in output_list:

        print(x)

    acc_score = accuracy_score(y_pred_r, y_test)
    print("Accuracy Score: ", acc_score)

    print("Classification Report: ")
    print(classification_report(y_pred_r, y_test))

    print("Confusion Matrix: ")
    print(confusion_matrix(y_pred_r, y_test))

    print("==================================================\n")

    return acc_score

train_network(700)