import face_recognition
from PIL import Image, ImageDraw

# 画像を読み込む
load_image = face_recognition.load_image_file("data/lena.jpg")

# 認識させたい画像から顔検出する
face_locations = face_recognition.face_locations(load_image)
pil_image = Image.fromarray(load_image)
pil_image.show()

im_back = pil_image.copy()
draw = ImageDraw.Draw(im_back)
print(type(face_locations))
print(face_locations)

for (top, right, bottom, left) in face_locations:
    draw.rectangle(((left, top), (right, bottom)), 
                   outline=(255, 0, 0), width=2)
    
im_back.show()

[(top, right, bottom, left)] = face_locations
"""
center = ((left + right)/2, (top + bottom)/2)
radius = max([right - center(0), bottom - center(1)])
"""
mask_im = Image.new("L", pil_image.size, 0)
draw = ImageDraw.Draw(mask_im)

draw.ellipse((top, right, bottom, left), fill=255)
mask_im.show()
