DeepImageSearch

#Image based search using VGG16 weights

#Input images features have been saved on output.hdf5

#Input images are in --dataset path, Query image is in --query

######################################################################


....extract_cnn_vgg16_keras.py : To extract feature using Imagenet weights from VGG16 architecture

....index.py : To write extracted features in hdf5 format for future use

....query_online : To search and find similar images 
