#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


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

training_generator = ImageDataGenerator(rescale = 1./255)
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
# Network Architecture
#########################################################################

# network architecture

model  = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same' , input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

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
model_filename = 'models/fruits_cnn_v01.h5' 


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

model_filename = 'models/fruits_cnn_v01.h5' 
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# load model

model = load_model(model_filename)

# import image & apply pre-processing

file_path = 'data/test/banana/banana_0074.jpg'

image = load_img(file_path, target_size = (img_width, img_height))
image = img_to_array(image)
image.shape   # (128, 128, 3)
image = np.expand_dims(image, axis = 0)
image.shape   # (1, 128, 128, 3) => 1 is for batch size as we only have 1 image
image = image * (1./255)  # Normalize

class_probs = model.predict(image) 
"""
array([[6.5489560e-03, 6.5369379e-05, 7.5541025e-01, 1.4614159e-03,
        2.1638182e-01, 2.0132143e-02]], dtype=float32)
"""

predicted_class = np.argmax(class_probs) # 2
predicted_lable = labels_list[predicted_class] # 'banana'
predicted_prob = class_probs[0][predicted_class] # 0.75541025 confident


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


image = preprocess_image(file_path)
make_prediction(image)



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


# overall test set accuracy 

prediction_df['correct'] = np.where(prediction_df['actual_label'] == prediction_df['predicted_label'], 1, 0)
test_set_accuracy = prediction_df['correct'].sum() / len(prediction_df)
print(test_set_accuracy) # 0.8166666666666667


# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(prediction_df['predicted_label'], prediction_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(prediction_df['predicted_label'], prediction_df['actual_label'], normalize = 'columns')
print(confusion_matrix)










