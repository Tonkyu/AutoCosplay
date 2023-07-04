from PIL import Image, ImageDraw, ImageFilter
#import matplotlib.pyplot as plt
import cv2
import dlib
import numpy as np
import sys
import re
import os

sys.path.insert(0, 'poisson-image-editing/')
from poisson_image_editing import poisson_edit

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def face_exchange(base, to, predictor):
    """
    base 顔を切り抜かれる方の画像のパス
    to 顔を貼り付けられる方の画像のパス
    返り値 新しく生成した画像のパス
    """
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    base_cv = cv2.imread(base)
    to_cv = cv2.imread(to)
    gray = cv2.cvtColor(base_cv, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    #　移す元の顔の倍率を測って，顔の大きさを移す先に合わせる．
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
    # 倍率測って移す元のサイズを変換
    vec_base_x = shape_base[16, :] - shape_base[0, :]
    vec_base_y = shape_base[27, :] - shape_base[8, :]
    vec_to_x = shape_to[16, :] - shape_to[0, :]
    vec_to_y = shape_to[27, :] - shape_to[8, :]
    ratio_x = np.linalg.norm(vec_to_x) / np.linalg.norm(vec_base_x)
    ratio_y = np.linalg.norm(vec_to_y) / np.linalg.norm(vec_base_y)
    basep = Image.open(base)
    size = [basep.size[0] * ratio_x, basep.size[1] * ratio_y]
    new_size = (np.array(size)).astype(int)
    base_resize = basep.resize(tuple(new_size))
    base_resize.save("images/base_resize.jpeg")

    # リサイズした画像の顔のぽちぽちを図る．
    base_cv = cv2.imread("images/base_resize.jpeg")
    gray = cv2.cvtColor(base_cv, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape_base = shape_np

    # 鼻の角度を図る
    vec_base = shape_base[27, :] - shape_base[30, :]
    vec_to = shape_to[27, :] - shape_to[30, :]
    theta = np.arccos(
        np.dot(vec_base, vec_to) / (np.linalg.norm(vec_base) * np.linalg.norm(vec_to))
    )
    #ratio = np.linalg.norm(vec_to) / np.linalg.norm(vec_base)
    #mat = ratio * np.array([

    # 移動分のベクトルを求める
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

    # マスク作り
    top = Image.open(to)
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
    im_blur = im.filter(ImageFilter.GaussianBlur(10)) # mask image

    base_resize = Image.open("images/base_resize.jpeg")
    mask_size = (max(im_blur.size[0], base_resize.size[0]),
                 max(im_blur.size[1], base_resize.size[1]))
    base_expand = Image.new(base_resize.mode, mask_size, (0, 0, 0))
    base_expand.paste(base_resize)
    base_rotate = base_expand.rotate(np.rad2deg(theta), center=tuple(shape_base[30, :]),
                                     translate=tuple(shape_to[30, :] - shape_base[30, :]))
    
    im_crop = base_rotate.crop((0, 0, top.size[0], top.size[1]))

    im_crop_cv2 = pil2cv(im_crop)
    top_cv2 = pil2cv(top)
    im_blur_cv2 = np.array(im)

    result = poisson_edit(im_crop_cv2, top_cv2, im_blur_cv2, offset=(0,0))

    p = re.compile("(.+)\..+")
    output_path = "../images/output_" \
        + p.findall(os.path.basename(base))[0] + '_' + p.findall(os.path.basename(to))[0] + ".jpeg"
    cv2.imwrite(output_path, result)
    return output_path

if __name__ == "__main__":
    import sys
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    output = face_exchange("lennon.jpeg", "megane.jpeg", predictor)
    img = cv2.imread(output)
    if img is None:
        sys.exit()

    cv2.imshow("win_img", img)
    cv2.waitKey(0)
    cv2.destroyWindow("win_img")