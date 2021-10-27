import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# using GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def read_file_for_data(path):
    with open(path) as f:
        text = f.readlines()
        text = [line.replace("\n", "") for line in text]
        data = np.array([line.split() for line in text])
    return data

test_dataset = read_file_for_data("./oxford-102-flowers/train.txt")
# test_dataset[:5]

cross_valid_dataset = read_file_for_data("./oxford-102-flowers/valid.txt")
# cross_valid_dataset[:5]

train_dataset = read_file_for_data("./oxford-102-flowers/test.txt")
# train_dataset[:5]

def generate_features_and_labels(dataset):
    images = np.array(dataset[:, 0])
    features = []

    for image in images:
        img = cv2.imread(f"./oxford-102-flowers/{image}")
        img = cv2.resize(img, (64, 64))
        features.append(img)

    features = np.array(features)
    features = features / 255.0
    labels = np.array([int(x) for x in dataset[:, 1]])

    return features, labels

x_train_1, y_train_1 = generate_features_and_labels(train_dataset[:1000])
# x_train, y_train

x_train_2, y_train_2 = generate_features_and_labels(train_dataset[1000:2000])
x_train4 = np.concatenate((x_train_1, x_train_2))
y_train4 = np.concatenate((y_train_1, y_train_2))

x_train_3, y_train_3 = generate_features_and_labels(train_dataset[2000:3000])
x_train = np.concatenate((x_train4, x_train_3))
y_train = np.concatenate((y_train4, y_train_3))

print(len(x_train), len(y_train))

with tf.device('/GPU:0'):
    model = Sequential([
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        #layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        #layers.MaxPooling2D(),
        # layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(102, activation="softmax")
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
