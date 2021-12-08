from __future__ import division
import random
import pprint
import keras
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from frcnn import data_generators
from frcnn import config
from frcnn import losses as losses
from frcnn import vgg
from frcnn.data_parser import get_data


# make dirs to save rpn
# "./models/rpn/rpn"
if not os.path.isdir("models"):
    os.mkdir("models")
if not os.path.isdir("models/rpn"):
    os.mkdir("models/rpn")


# pass the settings from the command line, and persist them in the config object
C = config.Config()

# set data argumentation
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

C.model_path = './model_frcnn.hdf5'
C.num_rois = 10

# we will use vgg
C.network = 'vgg16'

# set the path to weights based on backend and model
C.base_net_weights = vgg.get_weight_path()

# place weight files on your directory
base_net_weights = vgg.get_weight_path()

#### load images here ####
# get voc images
all_imgs, classes_count, class_mapping = get_data(C.train_path+'/_annotations.csv')
valid_imgs, valid_classes_count, valid_class_mapping = get_data(C.valid_path+'/_annotations.csv')

print(classes_count)

# add background class as 21st class
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = 'config.pickle'

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

# split to train and val
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, vgg.get_img_output_length,
                                               K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, vgg.get_img_output_length,
                                             K.image_dim_ordering(), mode='val')

# set input shape
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# create rpn model here
# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = vgg.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
# rpn outputs regression and cls
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = vgg.rpn(shared_layers, num_anchors)

model_rpn = Model(img_input, rpn[:2])

# load weights from pretrain
try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    #	model_classifier.load_weights(C.base_net_weights, by_name=True)
    print("loaded weights!")
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# compile model
optimizer = Adam(lr=1e-5, clipnorm=0.001)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_rpn.summary()

# write training misc here
epoch_length = 1000
num_epochs = 50
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

# start acutual training here
# X, Y, img_data = next(data_gen_train)
#
##loss_rpn = model_rpn.train_on_batch(X, Y)
# P_rpn = model_rpn.predict_on_batch(X)

# you should enable NMS when you visualize your results.
# NMS will filter out redundant predictions rpn gives, and will only leave the "best" predictions.
# P_rpn = model_rpn.predict_on_batch(image)
# R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
# X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
# this will output the binding box axis. [x1,x2,y1,y2].

Callbacks = keras.callbacks.ModelCheckpoint(
    "./models/rpn/rpn." + C.network + ".weights.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=1,
    save_best_only=True, save_weights_only=True, mode='auto', period=4)
callback = [Callbacks]
if len(val_imgs) == 0:
    # assuming you don't have validation data
    history = model_rpn.fit_generator(data_gen_train,
                                      epochs=num_epochs, steps_per_epoch=epoch_length,
                                      callbacks=callback)
    loss_history = history.history["loss"]
else:
    history = model_rpn.fit_generator(data_gen_train,
                                      epochs=num_epochs, validation_data=data_gen_val,
                                      steps_per_epoch=epoch_length, callbacks=callback, validation_steps=100)
    loss_history = history.history["val_loss"]

import numpy

numpy_loss_history = numpy.array(loss_history)
numpy.savetxt(C.network + "_rpn_loss_history.txt", numpy_loss_history, delimiter=",")
