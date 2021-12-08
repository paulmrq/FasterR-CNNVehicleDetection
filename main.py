import pandas as pd
import matplotlib.pyplot as plt
from frcnn.config import Config
import os


train_url = './data/train/'
test_url = './data/test/'
valid_url = './data/valid/'
base_path = '/kaggle/input'

train_path =  '/kaggle/input/processed/annotation.txt' # Training data (annotation file)

num_rois = 4 # Number of RoIs to process at once.

# Augmentation flag
horizontal_flips = True # Augment with horizontal flips in training.
vertical_flips = True   # Augment with vertical flips in training.
rot_90 = True           # Augment with 90 degree rotations in training.

output_weight_path = os.path.join('model_frcnn_vgg.hdf5')

record_path = os.path.join('record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

base_weight_path = os.path.join(base_path, 'models/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

config_output_filename = os.path.join('model_vgg_config.pickle')

def main():
    """ train = pd.read_csv(train_url + '_annotations.csv')
       test = pd.read_csv(test_url + '_annotations.csv')
       valid = pd.read_csv(valid_url + '_annotations.csv')

       # print(train.head())
       image = plt.imread(train_url + train.filename[0])
       plt.imshow(image)
       plt.show()

       print(train['class'].unique())"""

    C = Config()

    C.use_horizontal_flips = horizontal_flips
    C.use_vertical_flips = vertical_flips
    C.rot_90 = rot_90

    C.record_path = record_path
    C.model_path = output_weight_path
    C.num_rois = num_rois

    C.base_net_weights = base_weight_path



if __name__ == "__main__":
    main()