#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################




#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters



# image generators



# image flows



#########################################################################
# Network Architecture
#########################################################################

# network architecture



# compile network



# view network architecture




#########################################################################
# Train Our Network!
#########################################################################

# training parameters



# callbacks



# train the network



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



# parameters for prediction



# load model



# import image & apply pre-processing



# image pre-processing function



# image prediction function



# loop through test data


        
# create dataframe to analyse



# overall test set accuracy



# confusion matrix













