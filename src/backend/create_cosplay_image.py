import cv2
import dlib
import numpy as np
import sys
from os import path

sys.path.insert(0, 'poisson-image-editing/')

#from paint_mask import MaskPainter
#from move_mask import MaskMover
#from poisson_image_editing import poisson_edit


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
    for i, (x, y) in enumerate(shape):
      cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('Landmark Detection', image)
    cv2.waitKey()
    return shape

# ここ作る
def create_mask_image(human_image,anime_img):
    human_face_points = create_face_parts(human_image)
    anime_face_points = create_face_parts(anime_img)


    mask_image_path = ''

    original_mask_img
    return mask_image_path


def calculate_main_parts(face_parts):
    return 0

def merge_images(human_img, anime_img):
    mask_img_path = create_mask_image(human_img, anime_img)
    mask_img = cv2.imread(mask_img_path)
    result = poisson_edit(human_img, anime_img, mask_img, offset)
    result = ''
    return result


def create_cosplay_image(human_img_path, anime_img_path):
    human_img = cv2.imread(human_img_path)
    anime_img = cv2.imread(anime_img_path)

    merge_result = merge_images(human_img, anime_img)
    output_path = ''
    # cv2.imwrite(output_path, poisson_blend_result)


if __name__ == '__main__':
		human_img_path = '../images/初音ミク-1.png'
		anime_img_path = ''
		create_cosplay_image(human_img_path, anime_img_path)
