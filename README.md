# FasterR-CNNVehicleDetection
Deep Learning student project on vehicle detection. This code is huge due to the size of our weights and the datasets comitted for teacher rating.
Thi code was inspired from this [repository](https://github.com/kentaroy47/frcnn-from-scratch-with-keras/blob/master/test_frcnn.py) but only implicate a VGG16 support.

### Architecture
- **RPN with VGG16 as CNN backbone** 


- **VGG16 for predictor CNN backbone**


- **Data augmentation**


- **ROI pooling**
### Datasets
**Homemade**: merged datasets from [Roboflow](https://public.roboflow.com/object-detection/vehicles-openimages) and Internet. The train,test, valid datasets are comitted especially for student purpose and then not included in the .gitignore 

## Running scripts..
### 1. Requirements
#### a. Create environnement

Install VENV environnement, requirements.txt and get pretrain VGG16 weights by running the script inside the cloned repository:

    ./install.sh

#### b. Download pretrained weights.
Using imagenet pretrained VGG16 weights will significantly speed up training.
This shell script will create a pretrain directory and download the weights file to: 'pretrain/vgg16_weights_tf_dim_ordering_tf_kernels.h5'.


    # place VGG16 weights in pretrain dir.
    mkdir pretrain & mv pretrain
    wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

- Optionnal : I recommend using the pretrained RPN model if you want to recompute a full training, which will stablize training. You can download the rpn model (VGG16) from here: https://drive.google.com/file/d/1IgxPP0aI5pxyPHVSM2ZJjN1p9dtE4_64/view?usp=sharing.
Place those weights in 'train/models/rpn'.

## Training
    
    #sample training
    python ./train/train_frcnn.py 

    #training using rpn pretrained weights download earlier , default --rpn is none. We could pass our proper RPN weights trained with train_rpn.py but it was too long to compute.
    python ./train/train_frcnn.py --rpn models/rpn/voc.hdf5

    # add --load yourmodelpath if you want to resume training.
    python ./train/train_frcnn.py --load ./train/models/model_frcnn.hdf5

Training will automatically create weights at ./train/models/model_frcnn.hdf5 that we can use for testing.
## Testing

To use **our trained weights** (around 20 hours of computation) and start testing. First of all download of weights [here](https://drive.google.com/file/d/1uKbDegYfr2n-tS9xusD-NBY_ILiWzohp/view?usp=sharing) and place it here: /train/models/model_frcnn.hdf5

    #start testing, default test path (if --p none) is ./data/test/ for images ./data/input/ for video, default --load is none
    python ./test/test_frcnn.py --load ./train/models/model_frcnn.hdf5

If we want to predict on a video, we need to pass the arguments '--type_media video' and to fill the input directory in ./data/input with a mp4 file


###Sources: 
 - https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/

 - Faster R-CNN from scratch: https://github.com/kentaroy47/frcnn-from-scratch-with-keras/blob/master/test_frcnn.py

 - video with Faster R-CNN: https://github.com/riadhayachi/faster-rcnn-keras
