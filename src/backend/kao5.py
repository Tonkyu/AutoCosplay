from PIL import Image, ImageDraw, ImageFilter
#import matplotlib.pyplot as plt
import cv2
import dlib
import numpy as np
import re
import os

detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class My_image:
    def __init__(
            self, new_size, mask_size, theta,
            center, translate, top, im_blur,
            shape_to, detector, predictor
        ):
        self.new_size = new_size
        self.mask_size = mask_size
        self.theta = theta
        self.center = center
        self.translate = translate
        self.top = top
        self.im_blur = im_blur
        self.shape_to = shape_to
        self.detector = detector
        self.predictor = predictor

    def shape_process(self, shape_base, base_resize):
        vec_base = shape_base[27, :] - shape_base[30, :]
        vec_to = self.shape_to[27, :] - self.shape_to[30, :]
        theta = np.arccos(
            np.dot(vec_base, vec_to)/(np.linalg.norm(vec_base) * np.linalg.norm(vec_to))
        )

        mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        v = shape_base - shape_base[30, :]
        l = []
        for vec in v:
            l.append(mat @ vec)

        l = np.array(l)
        new_a = l + self.shape_to[30, :]
        new_a = new_a.astype(int)
        im = Image.new("L", self.top.size, 0)
        draw = ImageDraw.Draw(im)
        for i, v in enumerate(new_a):
            if i <= 16 or i >= 27:
                draw.rectangle((v[0], v[1], new_a[30, 0], new_a[30, 1]), fill=255)
            elif i <= 21:
                draw.rectangle((min(v[0], self.shape_to[i, 0]), min(v[1], self.shape_to[i, 1]), new_a[30, 0], new_a[30, 1]),
                            fill=255)
            else:
                draw.rectangle((max(v[0], self.shape_to[i, 0]), min(v[1], self.shape_to[i, 1]), new_a[30, 0], new_a[30, 1]),
                            fill=255)

        im_blur = im.filter(ImageFilter.GaussianBlur(10))
        mask_size = (max(im_blur.size[0], base_resize.size[0]),
                 max(im_blur.size[1], base_resize.size[1]))
        base_expand = Image.new(base_resize.mode, mask_size, (0, 0, 0))
        base_expand.paste(base_resize)
        base_rotate = base_expand.rotate(np.rad2deg(theta), center=tuple(shape_base[30, :]),
                                         translate=tuple(self.shape_to[30, :] - shape_base[30, :]))
        im_crop = base_rotate.crop((0, 0, self.top.size[0], self.top.size[1]))
        result_img = self.top.copy()
        result_img.paste(im_crop, (0, 0), self.im_blur)
        return result_img
    
    def base_process(self, basep):
        base_resize = basep.resize(tuple(self.new_size))

        gray = cv2.cvtColor(np.array(base_resize), cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 1)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape_np = np.zeros((68, 2), dtype=int)
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape_base = shape_np
        result_img = self.shape_process(shape_base, base_resize)
        return result_img

def face_exchange(base_array, to_path, predictor):
    """
    base 顔を切り抜かれる方のnumpy配列
    to 顔を貼り付けられる方の画像のパス
    返り値 新しく生成した画像のnumpy配列
    """
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    to_cv = cv2.imread(to_path)
    gray = cv2.cvtColor(base_array, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape_base = shape_np

    gray = cv2.cvtColor(to_cv, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape_to = shape_np
    
    vec_base_x = shape_base[16, :] - shape_base[0, :]
    vec_base_y = shape_base[27, :] - shape_base[8, :]
    vec_to_x = shape_to[16, :] - shape_to[0, :]
    vec_to_y = shape_to[27, :] - shape_to[8, :]
    ratio_x = np.linalg.norm(vec_to_x) / np.linalg.norm(vec_base_x)
    ratio_y = np.linalg.norm(vec_to_y) / np.linalg.norm(vec_base_y)
    basep = Image.fromarray(base_array)
    size = [basep.size[0] * ratio_x, basep.size[1] * ratio_y]
    new_size = (np.array(size)).astype(int)
    base_resize = basep.resize(tuple(new_size))

    gray = cv2.cvtColor(np.array(base_resize), cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape_base = shape_np

    vec_base = shape_base[27, :] - shape_base[30, :]
    vec_to = shape_to[27, :] - shape_to[30, :]
    theta = np.arccos(
        np.dot(vec_base, vec_to)/(np.linalg.norm(vec_base) * np.linalg.norm(vec_to))
    )

    mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    v = shape_base - shape_base[30, :]
    l = []
    for vec in v:
        l.append(mat @ vec)

    l = np.array(l)
    new_a = l + shape_to[30, :]
    new_a = new_a.astype(int)

    top = Image.open(to_path)
    im = Image.new("L", top.size, 0)
    draw = ImageDraw.Draw(im)
    for i, v in enumerate(new_a):
        if i <= 16 or i >= 27:
            draw.rectangle((v[0], v[1], new_a[30, 0], new_a[30, 1]), fill=255)
        elif i <= 21:
            draw.rectangle((min(v[0], shape_to[i, 0]), min(v[1], shape_to[i, 1]), new_a[30, 0], new_a[30, 1]),
                           fill=255)
        else:
            draw.rectangle((max(v[0], shape_to[i, 0]), min(v[1], shape_to[i, 1]), new_a[30, 0], new_a[30, 1]),
                           fill=255)

    im_blur = im.filter(ImageFilter.GaussianBlur(10))
    mask_size = (max(im_blur.size[0], base_resize.size[0]),
                 max(im_blur.size[1], base_resize.size[1]))
    base_expand = Image.new(base_resize.mode, mask_size, (0, 0, 0))
    base_expand.paste(base_resize)
    base_rotate = base_expand.rotate(np.rad2deg(theta), center=(center := tuple(shape_base[30, :])),
                                     translate=(translate := tuple(shape_to[30, :] - shape_base[30, :])))
    
    im_crop = base_rotate.crop((0, 0, top.size[0], top.size[1]))
    top.paste(im_crop, (0, 0), im_blur)
    process_set = My_image(
        new_size, mask_size, theta, center, translate, Image.open(to_path), im_blur, shape_to,
        detector, predictor
    )
    return np.array(top), process_set

if __name__ == "__main__":
    import sys
    cap = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    ret, src = cap.read()
    if not ret:
        print("Can not open camera")
        sys.exit()
    
    base_array = np.array(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    output_array, process_set = face_exchange(base_array, "lennon_large.jpeg", predictor)
    img = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
    #img = cv2.imread(output)

    cv2.imshow("win_img", img)
    cv2.waitKey(0)
    cv2.destroyWindow("win_img")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can not open camera")
        sys.exit()

    while True:
        ret, src = cap.read()
        if not ret:
            break

        img = process_set.base_process(Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)))
        cv2.imshow("win_src", cv2.flip(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 1))
        if cv2.waitKey(90) == 27:
            break

    cv2.destroyAllWindows()