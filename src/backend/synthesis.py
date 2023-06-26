#rembgライブラリによる背景削除
#事前にpipコマンドでrembgライブラリをインストールする必要あり

import rembg
from PIL import Image
import io


# 背景を削除したい画像のパス
###frontendでキャラの名前を選択するとここにパスが渡るようにしたい!###
input_image_path = './images/初音ミク.png'

# 背景画像のパス
###frontendで背景の名前をを選択するとここにパスが渡るようにしたい!###
background_image_path = './images/background.png'

# 背景を削除したい画像を読み込み
with open(input_image_path, 'rb') as f:
    input_image = f.read()

# 背景を削除
output_image = rembg.remove(input_image)

# 背景を削除した画像をPIL Imageオブジェクトとして読み込み
output_image = Image.open(io.BytesIO(output_image)).convert('RGBA')

# 背景画像を読み込み
background_image = Image.open(background_image_path).convert('RGBA')

#背景画像を人物画像のサイズに合わせてリサイズ
width, height = output_image.size
new_size=(width,height)
resized_background = background_image.resize(new_size)

# 背景を削除した画像と背景画像を合成
composite_image = Image.alpha_composite(resized_background, output_image)

# 合成した画像を保存
composite_image.save('./images/composite_image.png')
