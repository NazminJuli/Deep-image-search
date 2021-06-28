
import os
import h5py
import numpy as np

from extract_cnn_vgg16_keras import VGGNet

# directory where raw datasets are located
root_path = "clusters"

# features saved in hdf5 format
index = "output.hdf5"


def get_imglist(path):
    final_path = []
    for root, sub_dir, files in os.walk(path):
      for sub in sub_dir:
        path = os.path.join(root,sub)

        for f in os.listdir(path):
                if f.endswith('.jpg'):
                 final_path.append(os.path.join(path,f))
    return final_path


img_list = get_imglist(root_path)
#print("img_list", img_list)


print("         feature extraction starts")
print("--------------------------------------------------")
    
feats = []
names = []

model = VGGNet()
for i, img_path in enumerate(img_list):

    norm_feat = model.extract_feat(img_path)
    img_name = img_path

    feats.append(norm_feat)
    names.append(img_name)

feats = np.array(feats)

print("--------------------------------------------------")
print("      writing feature extraction results ...")
print("--------------------------------------------------")

h5f = h5py.File(index, 'w')
h5f.create_dataset('dataset_1', data = feats)
h5f.create_dataset('dataset_2', data = np.string_(names))
h5f.close()
