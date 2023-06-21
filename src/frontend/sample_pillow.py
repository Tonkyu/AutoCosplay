from PIL import Image, ImageFilter

im = Image.open("data/Blender_Suzanne1.jpg")
print(im.format, im.size, im.mode)
print(type(im.format))