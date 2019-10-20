from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
import imutils
import numpy as np
import pandas as pd
import argparse
import dlib
import cv2

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

cap = cv2.VideoCapture(0)


interested_body_part = ["mouth", "right_eyebrow",
                        "left_eyebrow", "right_eye", "left_eye", "nose"]

parser = argparse.ArgumentParser()
parser.add_argument('--cnn_weight', default="mmod_human_face_detector.dat",
                    help="path to pretrained face detection")
parser.add_argument('-p', '--shape_predictor', default='./shape_predictor_68_face_landmarks.dat',
                    help='pretrained weights for shape detection')
args = vars(parser.parse_args())

# HOG-based face detection
detector = dlib.get_frontal_face_detector()

# CNN-based face detection
# detector = dlib.cnn_face_detection_model_v1(args['cnn_weight'])
predictor = dlib.shape_predictor(args['shape_predictor'])


# create face aligner
fa = FaceAligner(predictor, desiredFaceWidth=224)

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)

        clone = image.copy()
        # cv2.putText(clone, body_part, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7, (0, 0, 255), 2)

        # body_part_shape = shape[FACIAL_LANDMARKS_IDXS[body_part][0]: FACIAL_LANDMARKS_IDXS[body_part][1]]
        for part_body in interested_body_part:
            part_body_shape = shape[FACIAL_LANDMARKS_IDXS[part_body]
                                    [0]: FACIAL_LANDMARKS_IDXS[part_body][1]]
            for (x, y) in part_body_shape:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # (x, y, w, h) = cv2.boundingRect(np.array(body_part_shape))
        # roi = clone[y: y+h, x: x+w]
        # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        cv2.imshow('Facial keypoint', clone)
    # cv2.imshow('Image', clone)
    if (cv2.waitKey(300) & 0xff == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
