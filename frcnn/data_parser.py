import cv2
import sys
import numpy as np


def get_data(input_path):
    """Parse the data from annotation file

    Args:
      input_path: annotation folder path

    Returns:
          all_data: list(filepath, width, height, list(bboxes))
          classes_count: dict{key:class_name, value:count_num}
              e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
          class_mapping: dict{key:class_name, value: idx}
              e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
      """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    i = 1

    with open(input_path+'_annotations.csv', 'r') as f:
        print('Parsing annotation files')

        # skip first line which is the column line
        next(f)
        for line in f:

            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            line_split = line.strip().split(',')


            # Make sure the info saved in annotation file matching the format (filename,width,height,class,xmin, ymin, xmax, ymax)
            # Note:
            #	One path_filename might has several classes (class_name)
            #	xmin, ymin, xmin, ymin are the pixel value of the origial image, not the ratio value
            #	(xmin, ymin) top left coordinates; (xmax, ymax) bottom right coordinates
            #   xmin,ymin-------------------
            #	|						|
            #	|						|
            #	|						|
            #	|						|
            #	---------------------xmax,ymax
            (filename, width, height, class_name, xmin, ymin, xmax, ymax) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(input_path+filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = width
                all_imgs[filename]['height'] = height
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'xmin': int(xmin), 'xmax': int(xmax), 'ymin': int(ymin), 'ymax': int(ymax)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping
