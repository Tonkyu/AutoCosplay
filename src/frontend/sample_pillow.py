from PIL import Image, ImageFilter, ImageDraw, ImageFont

im = Image.open("data/Blender_Suzanne1.jpg")
print(im.format, im.size, im.mode)
print(type(im.format))

# RGB各色の最小値と最大値を取得
print(im.getextrema())

print(im.getpixel((256, 256)))

# 白黒変換convert("L")，90度回転rotate(90)，ガウシアンブラー
new_im = im.convert("L").rotate(90).filter(ImageFilter.GaussianBlur())
new_im.show()

new_im.save("data/sample_pillow.jpg")

# ベタ塗り画像生成
im = Image.new("RGB", (512, 512), (128, 128, 128))
im.show()

# 図形の描画
draw = ImageDraw.Draw(im)

# 直線，長方形，楕円を描画
draw.line((0, im.height, im.width, 0), fill=(255, 0, 0), width=8)
draw.rectangle((100, 100, 200, 200), fill=(0, 255, 0))
draw.ellipse((250, 300, 450, 400), fill=(0, 0, 255))

font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 48)
draw.multiline_text((0, 0), "Pillow sample", fill=(0, 0, 0), font=font)
im.save("data/pillow_image_draw.jpg")
im.show()