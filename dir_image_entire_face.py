from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_IDXS, FaceAligner
import imutils
import numpy as np
import pandas as pd
import argparse
import dlib
import cv2
import os

import utils

# helper function

# FACIAL_LANDMARKS_IDXS = OrderedDict([
#     ("mouth", (48, 68)),
#     ("right_eyebrow", (17, 22)),
#     ("left_eyebrow", (22, 27)),
#     ("right_eye", (36, 42)),
#     ("left_eye", (42, 48)),
#     ("nose", (27, 35)),
#     ("jaw", (0, 17))
# ])


# global variable
image_extension = ['.jpg', '.jpeg', '.png']
interested_body_part = ["mouth", "right_eyebrow",
                        "left_eyebrow", "right_eye", "left_eye", "nose"]


# helper function
def save_aligned_face(aligned_face, root, raw_dir, save_dir, file_name, bb_index):
    ''' This function to save an aligned face according to the directory
    '''
    # manipulate aligned face path to save
    aligned_save_path = root + '/' + file_name[: file_name.rfind('.')]
    aligned_save_path = utils.change_main_directory(
        aligned_save_path, raw_dir, save_dir)
    if (bb_index != 0):
        aligned_save_path += '_' + str(bb_index)
    aligned_save_path += file_name[file_name.rfind('.'):]

    # save aligned face into the directory
    cv2.imwrite(aligned_save_path, aligned_face)

    print('Save finished : ', aligned_save_path)
    return aligned_save_path


def face2bb(rect):
    ''' This function to transform a face to a bounding box
    '''
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

    return (x, y, w, h)


def shape_detector(save_aligned_path, rect):
    body_data = dict()
    for interested_part in interested_body_part:
        body_data[interested_part] = list()

    aligned_face = cv2.imread(save_aligned_path)
    gray_aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray_aligned_face, rect)
    shape = face_utils.shape_to_np(shape)

    for part_body in interested_body_part:
        part_body_shape = shape[FACIAL_LANDMARKS_IDXS[part_body][0]:
                                FACIAL_LANDMARKS_IDXS[part_body][1]]

        count = 0
        # print('#'*50)
        for (x, y) in part_body_shape:

            body_data[part_body].append((x, y))

            # print('Position ' + str(count) +
            #       ' : ' + str(x) + ', ' + str(y))
            count += 1
            # cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        # print('#'*50)
    print(body_data)
    return body_data


parser = argparse.ArgumentParser()
parser.add_argument('--cnn_weight', default="mmod_human_face_detector.dat",
                    help="path to pretrained face detection")
parser.add_argument('-p', '--shape_predictor', default='./shape_predictor_68_face_landmarks.dat',
                    help='pretrained weights for shape detection')
parser.add_argument('-d', '--dir_image', required=True,
                    help='path to data directory')
parser.add_argument('-s', '--save_image', required=True,
                    help='path to save preprocessed data')
args = vars(parser.parse_args())

# HOG-based face detection
# detector = dlib.get_frontal_face_detector()

# CNN-based face detection
detector = dlib.cnn_face_detection_model_v1(args['cnn_weight'])
predictor = dlib.shape_predictor(args['shape_predictor'])

# create face aligner
fa = FaceAligner(predictor, desiredFaceWidth=224)

print('Start to open through data directory')
for (root, subdirs, file_names) in os.walk(args['dir_image']):

     # create directories according to the directory hierarchy
    for subdir in subdirs:
        save_root = utils.change_main_directory(
            root, args['dir_image'], args['save_image'])
        if (not os.path.exists(save_root + '/' + subdir)):
            os.makedirs(save_root + '/' + subdir)
            print('#### CREATE FOLDER : ' + save_root + '/' + subdir)

    # read the data
    if (len(file_names) != 0):

        # create save root path
        rootSaveDir = args["save_image"] + root[root.find('/'):]

        for file_name in file_names:
            if ('._' in file_name or '.DS_Store' in file_name):
                continue
            print('OPEN IMAGE : ' + root + '/' + file_name)
            image = cv2.imread(root + '/' + file_name)
            image = cv2.resize(image, (224, 224))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect the face
            faces = detector(gray, 2)

            # initialize the index of rectangles
            bb_index = 0

            for index, face in enumerate(faces):

                (x, y, w, h) = face2bb(face)

                aligned_face = fa.align(image, gray, face.rect)

                save_aligned_path = save_aligned_face(
                    aligned_face, root, args['dir_image'], args['save_image'], file_name, bb_index)

                shape_detector(save_aligned_path)

                bb_index += 1


# while True:
#     _, image = cap.read()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     rects = detector(gray, 1)

#     for (i, rect) in enumerate(rects):
#         shape = predictor(gray, rect)

#         shape = face_utils.shape_to_np(shape)

#         clone = image.copy()
#         # cv2.putText(clone, body_part, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#         #     0.7, (0, 0, 255), 2)

#         # body_part_shape = shape[FACIAL_LANDMARKS_IDXS[body_part][0]: FACIAL_LANDMARKS_IDXS[body_part][1]]
#         count = 0
#         for part_body in interested_body_part:
#             part_body_shape = shape[FACIAL_LANDMARKS_IDXS[part_body]
#                                     [0]: FACIAL_LANDMARKS_IDXS[part_body][1]]
#             print('Part Name :', part_body)
#             print('#########')
#             for (x, y) in part_body_shape:
#                 print('Position ' + str(count) +
#                       ' : ' + str(x) + ', ' + str(y))
#                 count += 1
#                 cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
#             print('#########')

#         # (x, y, w, h) = cv2.boundingRect(np.array(body_part_shape))
#         # roi = clone[y: y+h, x: x+w]
#         # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

#         cv2.imshow('Facial keypoint', clone)
#     # cv2.imshow('Image', clone)
#     if (cv2.waitKey(300) & 0xff == ord('q')):
#         break

# cap.release()
# cv2.destroyAllWindows()
