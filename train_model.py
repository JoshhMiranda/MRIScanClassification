# 1: Import dependencies 

# For data manipulation
import numpy as np 
# import pandas as pd 
from PIL import Image

# For data visualization
import matplotlib.pyplot as plt 
# import seaborn as sns

# Ingore the warnings
import warnings
warnings.filterwarnings('ignore')

# DL Libraries
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

# Other libraries
import os
import random

import mlflow
import mlflow.keras


#### train a dummy model to check mlflow versioning


# 2. Load data from directory:

BATCH_SIZE = 64

train_ds = keras.utils.image_dataset_from_directory(
    directory = r'C:/Users/Joshua Miranda/PythonPractice/Projects/brain_tumor_datasets/Training',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(256, 256),
)

val_ds = keras.utils.image_dataset_from_directory(
    directory = r'C:/Users/Joshua Miranda/PythonPractice/Projects/brain_tumor_datasets/Testing',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(256, 256),
)

classes = train_ds.class_names



# 3. define model architecture: (dummy architecture to speed up training)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# 4. Compile model:
model.compile(optimizer= Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# 5. Set the MLflow experiment name
mlflow.set_experiment("mri_scan_classification_v1")


# 6. Start an MLflow run
model_name = "mri_scan_dummy_model2"
EPOCHS = 3

with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    # Train the dummy model:
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)
    
    # Log metrics
    metrics = history.history
    for epoch, loss in enumerate(metrics['loss']):
        mlflow.log_metric("train_loss", loss, step=epoch)
    for epoch, val_loss in enumerate(metrics['val_loss']):
        mlflow.log_metric("val_loss", val_loss, step=epoch)


    # Log accuracy plot:
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.plot(history.history['accuracy'], color='red',label='Train')
    plt.plot(history.history['val_accuracy'], color='blue',label='Validation')
    plt.legend()
    # Save the plot to a file
    plot_filename = model_name+"_Accuracy_training_validation.png"
    plt.savefig(plot_filename)
    # Log the plot as an artifact
    mlflow.log_artifact(plot_filename)


    # Plotting the graph of Accuracy and Validation loss
    plt.title('Training Loss vs Validation Loss')
    plt.plot(history.history['loss'], color='red',label='Train')
    plt.plot(history.history['val_loss'], color='blue',label='Validation')
    plt.legend()
    # Save the plot to a file
    plot_filename = model_name+"_Loss_training_validation.png"
    plt.savefig(plot_filename)
    # Log the plot as an artifact
    mlflow.log_artifact(plot_filename)


    # Log the model
    mlflow.keras.log_model(model, model_name)

    print(f"Run ID: {run.info.run_id}")