import cv2
import dlib
import numpy as np
import sys
from kao import face_exchange
from os import path
from PIL import Image, ImageDraw, ImageFilter 
import matplotlib.pyplot as plt

sys.path.insert(0, 'poisson-image-editing/')

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
    # for i, (x, y) in enumerate(shape):
    #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    # cv2.imshow('Landmark Detection', image)
    # cv2.waitKey()
    return shape

def create_mask_image(human_img, anime_img):
    human_face_points = create_face_parts(human_img)
    anime_face_points = create_face_parts(anime_img)

    im = Image.new("L",tuple(reversed(human_img.shape[:2])), 0)
    draw = ImageDraw.Draw(im)

    for v in human_face_points:
        min_x = min(v[0], human_face_points[30, 0])
        max_x = max(v[0], human_face_points[30, 0])
        min_y = min(v[1], human_face_points[30, 1])
        max_y = max(v[1], human_face_points[30, 1])
        draw.rectangle((min_x, min_y, max_x, max_y), fill=255)
    plt.imshow(im, cmap="gray")
    mask_image_path = 'mask.jpg'
    im.save(mask_image_path)
    return mask_image_path


def merge_images(human_img, anime_img, output_path):
    mask_img_path = create_mask_image(human_img, anime_img)
    mask_img = cv2.imread(mask_img_path)
    # result = poisson_edit(human_img, anime_img, mask_img, offset=(0,0))
    # cv2.imwrite(output_path, result)


def create_cosplay_image(human_img_path, anime_img_path, character_name):
    human_img = cv2.imread(human_img_path)
    anime_img = cv2.imread(anime_img_path)

    output_path = '../images/products/' + character_name
    merge_images(human_img, anime_img, output_path)
    return output_path

if __name__ == '__main__':
    human_img_path = '../images/characters/ace.png'
    anime_img_path = '../images/characters/miku.png'
    character_name = 'miku'
    # create_cosplay_image(human_img_path, anime_img_path, character_name)
    face_exchange(human_img_path, anime_img_path)
