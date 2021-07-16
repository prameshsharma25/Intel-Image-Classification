import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet_v2 import ResNet50V2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

train_path = '../input/intel-image-classification/seg_train/seg_train'
test_path = '../input/intel-image-classification/seg_test/seg_test'

# Hyperparameter
epochs = 10
batch_size = 32
target_size = (224, 224)

# Preprocess Data
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255
)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=target_size,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=target_size,
    validation_split=0.2,
    subset='validation',
    seed=42
)

test_dataset = data_generator.flow_from_directory(
    test_path,
    shuffle=False,
    seed=42
)

# Data labels
labels = {i:label for i, label in enumerate(train_dataset.class_names)}
num_classes = len(train_dataset.class_names)

fig = plt.figure(figsize=(8, 8))
for image, label in train_dataset.take(1):
    for i in range(6):
        plt.subplot(3, 2, i+1)
        img = np.array(image[i]/255)
        plt.imshow(img)
        plt.title(labels[int(label[i])])
        plt.axis("off")

# Perform data augmentation
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.Rescaling(1./255),
#         keras.layers.experimental.preprocessing.RandomContrast(0.3),
#         keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=target_size+(3,)),
#         keras.layers.experimental.preprocessing.RandomWidth(factor=(0.2, 0.3)),
#         keras.layers.experimental.preprocessing.RandomHeight(factor=(0.2, 0.3)),
#         keras.layers.experimental.preprocessing.RandomZoom(width_factor=(-0.2, -0.1), height_factor=(-0.2, -0.1))
    ]
)

# Train model
base_model = ResNet50V2(input_shape=target_size + (3,), include_top=False, weights='imagenet', pooling='max')
base_model.trainable = False

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

model = make_model(target_size + (3,), num_classes)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Model visualization
df = pd.DataFrame(history.history)
df[['loss', 'val_loss']].plot()
df[['accuracy', 'val_accuracy']].plot()

model.save('intel-image-cnn.h5')

predictions = model.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_dataset, predictions)