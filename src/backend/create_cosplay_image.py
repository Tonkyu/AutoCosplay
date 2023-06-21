import cv2
import dlib
import numpy as np

import sys
from os import path

sys.path.insert(0, 'poisson-image-editing/')

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit


def create_face_parts(image):

  # use dlib
  # reference: https://towardsdatascience.com/face-landmark-detection-using-python-1964cb620837
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 1)

  for rect in rects:
      shape = predictor(gray, rect)
      shape_np = np.zeros((68, 2), dtype="int")
      for i in range(68):
          shape_np[i] = (shape.part(i).x, shape.part(i).y)
      shape = shape_np
  return shape

def create_mask_image(face_points):
    mask_image_path = ''
    nose_point = (0, 0)
    return mask_image_path, nose_point


def calculate_offset(anime_nose_point, mask_nose_point):
    return (0, 0)


def create_cosplay_image(human_img_path, anime_img_path):
    human_img = cv2.imread(human_img_path)
    anime_img = cv2.imread(anime_img_path)
    face_points = create_face_parts(human_img)
    print(face_points)
    mask_image_path, mask_nose_point = create_mask_image(face_points)
    mask_img = cv2.imread(mask_image_path)
    anime_nose_point = (0, 0)
    offset = calculate_offset(anime_nose_point, mask_nose_point)

    poisson_blend_result = poisson_edit(human_img, anime_img, mask_img, offset)

    output_path = ''
    cv2.imwrite(output_path, poisson_blend_result)


if __name__ == '__main__':
		human_img_path = 'resources/input/fuku.jpg'
		anime_img_path = ''
		create_cosplay_image(human_img_path, anime_img_path)
