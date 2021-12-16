from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import itertools
import operator
import keras
import datetime
import re
import subprocess

from frcnn import config
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from frcnn import roi_helpers
from keras.applications.mobilenet import preprocess_input

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default='./data/test/')
#parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default='./data/input/')

parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg.",
                  default='vgg')
parser.add_option("--write", dest="write", help="to write out the image with detections or not.", action='store_true',
                  default=True)
parser.add_option("--load", dest="load", help="specify model path.", default=None)
parser.add_option("--type_media", dest="media", help="specify media type: video or image.", default='image')


(options, args) = parser.parse_args()

if not options.test_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

img_path = options.test_path

input_video_file = './data/input/input.mp4'
output_video_file = 'output.mp4'
output_path = 'output/'
frame_rate = 25.0


def cleanup():
    print("cleaning up...")
    os.popen('rm -f ' + input_video_file + '*')


def get_file_names(search_path):
    for (dirpath, _, filenames) in os.walk(search_path):
        for filename in filenames:
            yield filename  # os.path.join(dirpath, filename)


def convert_to_images():
    cam = cv2.VideoCapture(input_video_file)
    counter = 0
    while True:
        flag, frame = cam.read()
        if flag:
            cv2.imwrite(os.path.join(img_path, str(counter) + '.jpg'), frame)
            counter = counter + 1
        else:
            break
        if cv2.waitKey(1) == 27:
            break
        # press esc to quit
    cv2.destroyAllWindows()


def save_to_video():

    img0 = cv2.imread(os.path.join(output_path, '0.jpg'))
    height, width, layers = img0.shape

    # fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.cv.CV_FOURCC(*'XVID')
    videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    list_files = sorted(get_file_names(output_path),
                        key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for f in list_files:
        print("saving..." + f)
        img = cv2.imread(os.path.join(output_path, f))
        videowriter.write(img)
    videowriter.release()
    cv2.destroyAllWindows()


def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, sum(item[1] for item in subiter)


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2




sys.setrecursionlimit(40000)

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# we will use resnet. may change to vgg
if options.network == 'vgg':
    C.network = 'vgg16'
    from frcnn import vgg as vgg

else:
    print('Not a valid model')
    raise ValueError

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

C.model_path = 'train/models/model_frcnn.hdf5'


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

# may need to fix this up with your backbone..!
print("backbone is vgg. number of features chosen is 512")
num_features = 512

if keras.backend.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (VGG)
shared_layers = vgg.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = vgg.rpn_layer(shared_layers, num_anchors)

classifier = vgg.classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)

# model loading
if options.load == None:
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)
else:
    print('Loading weights from {}'.format(options.load))
    model_rpn.load_weights(options.load, by_name=True)
    model_classifier.load_weights(options.load, by_name=True)

model_rpn.compile(optimizer='adam', loss='mse')
model_classifier.compile(optimizer='adam', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5

visualise = True

num_rois = C.num_rois

if options.media == 'image':
    for idx, img_name in enumerate(sorted(os.listdir(img_path))):

        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join(img_path, img_name)

        img = cv2.imread(filepath)
        X,ratio = format_img(img, C)

        # preprocess image
        # X, ratio = format_img(img, C)
        img_scaled = (np.transpose(X[0, :, :, :], (1, 2, 0)) + 127.5).astype('uint8')

        if keras.backend.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, keras.backend.image_dim_ordering(), overlap_thresh=0.3)
        print(R.shape)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0] // num_rois + 1):
            ROIs = np.expand_dims(R[num_rois * jk:num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            print(P_cls)

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < 0.8 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                print(key)
                print(len(bboxes[key]))
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                            overlap_thresh=0.3)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk, :]
                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                                  (int(class_to_color[key][0]), int(class_to_color[key][1]),
                                   int(class_to_color[key][2])), 2)

                    textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                    all_dets.append((key, 100 * new_probs[jk]))

                    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    textOrg = (real_x1, real_y1 - 0)

                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            print('Elapsed time = {}'.format(time.time() - st))
            print(all_dets)
            print(bboxes)
            # enable if you want to show pics
            if options.write:
                import os

                if not os.path.isdir("output"):
                    os.mkdir("output")

                cv2.imwrite('./output/{}.png'.format(idx), img)

elif options.media == 'video':
    img_path = './data/input/'
    print("Converting video to images..")
    convert_to_images()
    print("anotating...")
    cleanup()

    list_files = sorted(get_file_names(img_path))

    for img_name in list_files:
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join(img_path, img_name)

        img = cv2.imread(filepath)
        X, ratio = format_img(img, C)

        # preprocess image

        img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
        img_scaled[:, :, 0] += 123.68
        img_scaled[:, :, 1] += 116.779
        img_scaled[:, :, 2] += 103.939

        img_scaled = img_scaled.astype('uint8')

        if keras.backend.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, keras.backend.image_dim_ordering(), overlap_thresh=0.3)
        print(R.shape)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0] // num_rois + 1):
            ROIs = np.expand_dims(R[num_rois * jk:num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            print(P_cls)

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < 0.8 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass

                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []
            all_objects = []

            for key in bboxes:
                print(key)
                print(len(bboxes[key]))
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                            overlap_thresh=0.3)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk, :]
                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                                  (int(class_to_color[key][0]), int(class_to_color[key][1]),
                                   int(class_to_color[key][2])),
                                  2)

                    textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                    all_dets.append((key, new_probs[jk]))
                    all_objects.append((key, 1))

                    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    textOrg = (real_x1, real_y1 - 0)

                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            print('Elapsed time = {}'.format(time.time() - st))
            height, width, channels = img_scaled.shape
            cv2.rectangle(img_scaled, (0, 0), (width, 30), (0, 0, 0), -1)
            cv2.putText(img_scaled, "Obj count: " + str(list(accumulate(all_objects))), (5, 19),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (255, 255, 255), 1)
            cv2.imwrite(os.path.join(output_path, img_name), img_scaled)
            print(all_dets)

        print("saving to video..")
        save_to_video()







