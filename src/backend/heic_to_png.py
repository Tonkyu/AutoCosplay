from PIL import Image

def convert_heic_to_png(heic_path, png_path):
    image = Image.open(heic_path)
    image.save(png_path, "PNG")

# 使用例
convert_heic_to_png("input.heic", "output.png")
