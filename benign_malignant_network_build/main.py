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
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,BatchNormalization, Dropout    # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator                          # type: ignore
from tensorflow.keras.models import Sequential                                               # type: ignore
from tensorflow.keras.callbacks import EarlyStopping


source_folder_parent = r"D:\VSC Workspace\breast_cancimg\jpeg"

dicom_info = pd.read_csv(r"D:\VSC Workspace\breast_cancimg\csv\dicom_info.csv")
mass_case_desc_test_set = pd.read_csv(r"D:\VSC Workspace\breast_cancimg\csv\mass_case_description_test_set.csv")
mass_case_desc_train_set = pd.read_csv(r"D:\VSC Workspace\breast_cancimg\csv\mass_case_description_train_set.csv")
meta = pd.read_csv(r"D:\VSC Workspace\breast_cancimg\csv\meta.csv")
calc_case_desc_test_set = pd.read_csv(r"D:\VSC Workspace\breast_cancimg\csv\calc_case_description_test_set.csv")
calc_case_desc_train_set = pd.read_csv(r"D:\VSC Workspace\breast_cancimg\csv\calc_case_description_train_set.csv")

DIMENSION = 200

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
                    print(image_path)
                    shutil.copy(image_path, "breast_cancer_CNN" + "\\" + train_or_test + "\\" + image_type + "_files" + "\\" + pathology + "\\" + image)

        else:

            for image in source_dir:

                if (image[0] == "1"):

                    image_path = source_folder + "\\" + image
                    print(image_path)

                    shutil.copy(image_path, "breast_cancer_CNN" + "\\" + train_or_test + "\\" + image_type + "_files" + "\\" + pathology + "\\" + image)


def train_network(X_train, X_test, DIMENSION, train_path, test_path, classes_num, end_activation):

    datagen = ImageDataGenerator(

        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 40

    )

    train_dataset = datagen.flow_from_directory(train_path, class_mode = "binary")
    test_dataset = datagen.flow_from_directory(test_path, class_mode = "binary")

    y_train = train_dataset.classes
    y_test = test_dataset.classes

    model=Sequential()

    model.add(Conv2D(128,(3,3),activation="relu",input_shape=(DIMENSION, DIMENSION, 3)))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),activation="relu"))

    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation = "relu"))

    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())

    model.add(Dense(64, activation = "relu"))

    model.add(Dense(8, activation = "relu"))
    
    model.add(Dense(classes_num, activation = end_activation))

    model.summary()

    model.compile(optimizer="adam", loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=5)

    history=model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=30, callbacks = [early_stop], shuffle=True)

    y_pred = model.predict(X_test)

    print("Y_pred: ", y_pred)

    y_pred_r = np.argmax(y_pred, axis = 1) 

    print("Y_pred_r: ", y_pred_r)

    #output_list = list(zip(y_pred_r, y_test))
    #for x in output_list:

    #    print(x)

    acc_score = accuracy_score(y_pred_r, y_test)
    print("Accuracy Score: ", acc_score)

    print("Confusion Matrix: ")
    print(confusion_matrix(y_pred_r, y_test))

    plt.plot(history.history['accuracy'], label = 'accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc = 'lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Test Accuracy == ", test_acc)

    return model



train_path6 = r"D:\VSC Workspace\combined_dataset\train\cropped_files"
test_path6 = r"D:\VSC Workspace\combined_dataset\test\cropped_files"

benign_train_path6 = r"D:\VSC Workspace\combined_dataset\train\cropped_files\BENIGN"
benign_without_callback_train_path6 = r"D:\VSC Workspace\combined_dataset\train\cropped_files\BENIGN_WITHOUT_CALLBACK"
malignant_train_path6 = r"D:\VSC Workspace\combined_dataset\train\cropped_files\MALIGNANT"

benign_test_path6 = r"D:\VSC Workspace\combined_dataset\test\cropped_files\BENIGN"
benign_without_callback_test_path6 = r"D:\VSC Workspace\combined_dataset\test\cropped_files\BENIGN_WITHOUT_CALLBACK"
malignant_test_path6 = r"D:\VSC Workspace\combined_dataset\test\cropped_files\MALIGNANT"

benign_train6 = load_images(benign_train_path6, DIMENSION)
benign_without_callback_train6 = load_images(benign_without_callback_train_path6, DIMENSION)
malignant_train6 = load_images(malignant_train_path6, DIMENSION)

benign_test6 = load_images(benign_test_path6, DIMENSION)
benign_without_callback_test6 = load_images(benign_without_callback_test_path6, DIMENSION)
malignant_test6 = load_images(malignant_test_path6, DIMENSION)


X_train6 = benign_train6 + malignant_train6 + benign_without_callback_train6
X_test6 = benign_test6 + malignant_test6 + benign_without_callback_test6

X_train6 = np.array(X_train6)
X_train6 = X_train6 / 255.0

X_test6 = np.array(X_test6)
X_test6 = X_test6 / 255.0


model6 = train_network(X_train6, X_test6, DIMENSION, train_path6, test_path6, 3, "relu")
