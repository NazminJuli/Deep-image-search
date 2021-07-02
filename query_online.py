
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import cv2


# directory where datasets are located
#result = "dataset/queries"

# directory where query(input) .jpg image is located
query_dir = "  "

# directory to save/write similar images (optional)
image_write_path = "  "

# hdf5 file to save output image name (optional)
matched_image_file = "match_images.hdf5"

index = "output.hdf5"    # features saving in hdf5 format
similar_score = 0.65    # define percentage of similarity 65% or more

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(index,'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
#print(feats)

imgNames = h5f['dataset_2'][:]
#print("imagename :",imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")
    
# read and show query image
queryDir = query_dir
queryImg = cv2.imread(queryDir)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute similarity score and sorting array
queryVec = model.extract_feat(query_dir)
scores = np.dot(queryVec, feats.T)

rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

#print("rank id",rank_ID)
print("rank score",rank_score)

limit = 0 # To count the number of similar sorted image based on score
len_score = len(rank_score)
print("length:", len_score)

for i in range(len_score):

    if rank_score[i] >= similar_score:
     limit = limit+1

    else:
     print("similar score minimum is",similar_score)
     break

print("--------------------------------------------------")
print("      writing matched images name with path ...")
print("--------------------------------------------------")

imagelist = [imgNames[index] for i, index in enumerate(rank_ID[0:limit])]
print("top matched (", limit, ") images in order are:", imagelist)
# write
h5f_output = h5py.File(matched_image_file, 'w')
h5f_output.create_dataset('dataset_1', data = imagelist)

# # read
# h5f_output = h5py.File(matched_image_file, 'r')
# imgNames_output = h5f_output['dataset_1'][:]
# print("imagename :",imgNames_output)
# h5f.close()

h5f.close()


# show top #limit number of retrieved result in image_write_path
for i, im in enumerate(imagelist):

    path = str(im, 'utf-8')
    print("image path matched:",path)
    image = cv2.imread(path)
    drive, filename = os.path.split(path)

    cv2.imwrite(image_write_path + "/"+ filename, image)
    # plt.title("search output %d" %(i+1))
    # plt.imshow(image)
    # plt.show()


