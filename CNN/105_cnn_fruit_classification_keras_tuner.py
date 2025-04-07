
#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import os


#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6



# image generators

training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range=(0.5, 1.5),
                                        fill_mode = 'nearest')
                                        
validation_generator = ImageDataGenerator(rescale = 1./255)



# image flows

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

#########################################################################
# Architecture Tuning
#########################################################################

def build_model(hp):
    model  = Sequential()
    
    model.add(Conv2D(filters = hp.Int("Input_Conv_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3, 3), padding = 'same' , input_shape = (img_width, img_height, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    for i in range( hp.Int("n_Conv_Layers", min_value = 1, max_value = 3, step = 1)):
        
        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_filter", min_value = 32, max_value = 128, step = 32), kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    for j in range( hp.Int("n_Dens_Layers", min_value = 1, max_value = 4, step = 1)):
    
        model.add(Dense(hp.Int(f"Dense_{j}_Neurons", min_value = 32, max_value = 128, step = 32)))
        model.add(Activation('relu'))
        
        if hp.Boolean("Dropout"):
            model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    
    # compile network
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice("Optimizer", values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
    return model



tuner = RandomSearch(hypermodel= build_model,
                     objective= 'val_accuracy',
                     max_trials= 3,
                     executions_per_trial= 2,
                     directory= os.path.normpath('C:/'),
                     project_name= 'fruit-cnn',
                     overwrite= True)

tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 5,
             batch_size = 32)

# top networks
tuner.results_summary()

# best network - hyperparameters
tuner.get_best_hyperparameters()[0].values
"""
{'Input_Conv_Filters': 96,
 'n_Conv_Layers': 1,
 'Conv_0_filter': 64,
 'n_Dens_Layers': 1,
 'Dense_0_Neurons': 128,
 'Dropout': True,
 'Optimizer': 'RMSProp'}
"""

# summary of best network architecture
tuner.get_best_models()[0].summary()
"""
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 96)      2688      
_________________________________________________________________
activation (Activation)      (None, 128, 128, 96)      0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 96)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 64)        55360     
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 65536)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               8388736   
_________________________________________________________________
activation_2 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 774       
_________________________________________________________________
activation_3 (Activation)    (None, 6)                 0         
=================================================================
Total params: 8,447,558
Trainable params: 8,447,558
Non-trainable params: 0
"""


#########################################################################
# Network Architecture
#########################################################################

# network architecture

model  = Sequential()

model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding = 'same' , input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))




# compile network

model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])


# view network architecture

model.summary()


#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epochs = 50
model_filename = 'models/fruits_cnn_tuned.h5' 


# callbacks

save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)



# train the network

history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])



#########################################################################
# Visualise Training & Validation Performance
#########################################################################

import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])


#########################################################################
# Make Predictions On New Data (Test Set)
#########################################################################

# import required packages

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir


# parameters for prediction

model_filename = 'models/fruits_cnn_tuned.h5' 
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# load model

model = load_model(model_filename)

# image pre-processing function

def preprocess_image(file_path):
    
    image = load_img(file_path, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)  
    
    return image


# image prediction function

def make_prediction(image):
    
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs) 
    predicted_lable = labels_list[predicted_class] 
    predicted_prob = class_probs[0][predicted_class] 
    
    return predicted_lable, predicted_prob 



# loop through test data

source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []
predicter_probabilities = []
file_names = []

for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images:
        
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, pred_probability  = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicter_probabilities.append(pred_probability)
        file_names.append(image)
        
        
# create dataframe to analyse

prediction_df = pd.DataFrame({"actual_label": actual_labels,
                              "predicted_label": predicted_labels,
                              "predicted_probability": predicter_probabilities,
                              "filename": file_names})

prediction_df['correct'] = np.where(prediction_df['actual_label'] == prediction_df['predicted_label'], 1, 0)

# overall test set accuracy 

test_set_accuracy = prediction_df['correct'].sum() / len(prediction_df)
print(test_set_accuracy) 


# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(prediction_df['predicted_label'], prediction_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(prediction_df['predicted_label'], prediction_df['actual_label'], normalize = 'columns')
print(confusion_matrix)












