def create_cosplay_image(input_image_path, anime_image_path):
    output_image_path = ""
    return output_image_path

import cv2
import dlib
import numpy as np

import getopt
import sys
from os import path

def create_face_parts(image):
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

def synthesize_images(human_img_path, anime_img_path):
		human_img = cv2.imread(human_img_path)
		anime_img = cv2.imread(anime_img_path)
		shape = create_face_parts(human_img)
		print(shape)


if __name__ == '__main__':
		human_img_path = 'resources/input/fuku.jpg'
		anime_img_path = ''
		synthesize_images(human_img_path, anime_img_path)
