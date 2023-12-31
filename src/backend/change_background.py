#事前にpipコマンドでrembgライブラリをインストールする必要あり

import rembg
from PIL import Image
import io

def change_background(img1,img2):
  # 背景を削除したい画像のパス
  input_image_path = img1

  # 背景画像のパス
  background_image_path = img2

  with open(input_image_path, 'rb') as f:
      input_image = f.read()

  # 背景を削除
  output_image = rembg.remove(input_image)

  output_image = Image.open(io.BytesIO(output_image)).convert('RGBA')

  background_image = Image.open(background_image_path).convert('RGBA')

  #背景画像を人物画像のサイズに合わせてリサイズ
  width, height = output_image.size
  new_size=(width,height)
  resized_background = background_image.resize(new_size)

  # 背景を削除した画像と背景画像を合成
  composite_image = Image.alpha_composite(resized_background, output_image)

  # 合成した画像を保存
  composite_image.save('../images/uploaded/composite_image.png')
