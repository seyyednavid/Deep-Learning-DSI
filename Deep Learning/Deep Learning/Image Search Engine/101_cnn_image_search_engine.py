
#########################################################################
# Convolutional Neural Network - Image Search Engine
#########################################################################


###########################################################################################
# import packages
###########################################################################################



###########################################################################################
# bring in pre-trained model (excluding top)
###########################################################################################

# image parameters



# network architecture



# save model file




###########################################################################################
# preprocessing & featurising functions
###########################################################################################





###########################################################################################
# featurise base images
###########################################################################################

# source directory for base images



# empty objects to append to



# pass in & featurise base image set



# save key objects for future use



        
###########################################################################################
# pass in new image, and return similar images
###########################################################################################

# load in required objects



# search parameters


        
# preprocess & featurise search image


        
# instantiate nearest neighbours logic



# apply to our feature vector store



# return search results for search image (distances & indices)



# convert closest image indices & distances to lists



# get list of filenames for search results



# plot results

plt.figure(figsize=(12,9))
for counter, result_file in enumerate(search_result_files):    
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1)
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





