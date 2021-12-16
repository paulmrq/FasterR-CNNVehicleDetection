from __future__ import division
import random
import pprint
import sys
from optparse import OptionParser

import keras
import time
import numpy as np
import pickle
import os

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from frcnn import data_generators
from frcnn import config
from frcnn import losses as losses
from frcnn import vgg
from frcnn.data_parser import get_data
from frcnn.roi_helpers import calc_iou,rpn_to_roi

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--rpn", dest="rpn_weight_path", help="Input path for rpn.", default=None)
#parser.add_option("--rpn", dest="rpn_weight_path", help="Input path for rpn.", default=None)
parser.add_option("--load", dest="load", help="What model to load", default=None)

(options, args) = parser.parse_args()

# pass the settings from the command line, and persist them in the config object
C = config.Config()

# set data argumentation
C.use_horizontal_flips = True
C.use_vertical_flips = True
C.rot_90 = True

# input './train/models/vgg16/model_frcnn_vgg.hdf5'
C.model_path = options.load
C.num_rois = 10

# we will use vgg
C.network = 'vgg16'

# set the path to weights based on backend and model
C.base_net_weights = vgg.get_weight_path()

# place weight files on your directory
base_net_weights = vgg.get_weight_path()

#### load images here ####
# get voc images
all_imgs, classes_count, class_mapping = get_data(C.train_path)
# valid_imgs, valid_classes_count, valid_class_mapping = get_data(C.valid_path+'/_annotations.csv')

print(classes_count)
# mkdir to save models.
if not os.path.isdir("train/models"):
  os.mkdir("train/models")
if not os.path.isdir("train/models/"+C.network):
  os.mkdir(os.path.join("train/models", C.network))

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

data_gen_train = data_generators.get_anchor_gt(C.train_path, train_imgs, C, vgg.get_img_output_length, mode='train')
data_gen_val = data_generators.get_anchor_gt(C.train_path, val_imgs, C, vgg.get_img_output_length, mode='val')

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
rpn = vgg.rpn_layer(shared_layers, num_anchors)

classifier = vgg.classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# load weights from pretrain
try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
    print("VGG weights loaded")
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# may use this to resume from rpn models or previous training. specify either rpn or frcnn model to load
if options.load is not None:
    try:
        print("loading previous model from ", C.model_path )
        model_rpn.load_weights(C.model_path , by_name=True)
        model_classifier.load_weights(C.model_path , by_name=True)
        print(C.model_path+' loaded')
    except:
        print('Could not load model weights.')
elif options.rpn_weight_path is not None:
    print("loading RPN weights from ", options.rpn_weight_path)
    try:
        model_rpn.load_weights(options.rpn_weight_path, by_name=True)
        print("RPN weights loaded")
    except:
        print('Could not load pretrained RPN weights.')

else:
    print("no previous model was loaded")

# optimizer setup
optimizer = Adam(learning_rate=1e-5)
optimizer_classifier = Adam(learning_rate=1e-5)

# compile model after loading weights
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_rpn.summary()
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# write training misc here
epoch_length = 100
num_epochs = 1
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):
    progbar = Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    # first 3 epoch is warmup
    if epoch_num < 3 and options.rpn_weight_path is not None:
        K.set_value(model_rpn.optimizer.lr, 1e-3 / 30)
        K.set_value(model_classifier.optimizer.lr, 1e-3 / 3)

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:

                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            X, Y, img_data = next(data_gen_train)
            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)
            R = rpn_to_roi(P_rpn[0], P_rpn[1], C, keras.backend.image_dim_ordering(), use_regr=True, overlap_thresh=0.4,
                                       max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3])),
                            ("average number of objects", len(selected_pos_samples))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights('train/models/model_frcnn.hdf5')

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')
