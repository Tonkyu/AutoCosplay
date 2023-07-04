# streamlit run app.pyというコマンドをターミナルで実行して表示されるURLに以下を構築
import streamlit as st
import cv2
import numpy as np
import sys
import io
import os
import tempfile
from PIL import Image
import dlib

st.set_page_config(
    page_title="AutoCos",
    layout="wide",
)

def render()->st:
    bg_img = '''
    <style>
    .stApp {
      background-image: url("https://drive.google.com/uc?export=view&id=1hKAfLRL-SosYZHREqfqdoKFFNCOs1kDI");
      background-size: cover;
      background-repeat: no-repeat;
    }
    </style>
    '''

    return st.markdown(bg_img, unsafe_allow_html=True)

render()

# サイドバーレイアウト (Sidebar)
character_directory="../images/output"
character_list = [file for file in os.listdir(character_directory) if not file.startswith('.')]
character_list_without_extension = [os.path.splitext(filename)[0] for filename in character_list]
character_selected = st.sidebar.selectbox('コスプレできるキャラ一覧', character_list_without_extension)
character_path = os.path.join(character_directory, character_selected + ".jpeg")
if character_selected is not None:
    image = Image.open(character_path)
    st.sidebar.image(image,  use_column_width=True)


background_directory="../images/backgrounds"
background_list = [file for file in os.listdir(background_directory) if not file.startswith('.')]
background_list_without_extension = [os.path.splitext(filename)[0] for filename in background_list]
background_selected = st.sidebar.selectbox('変更できる背景一覧', background_list_without_extension,key="background_select1")
background_path = os.path.join(background_directory, background_selected + ".png")

if background_selected is not None:
    image = Image.open(background_path)
    st.sidebar.image(image,  use_column_width=True)

markdown = """
# プロジェクト演習E班 成果物
##  どこでもコスプレアプリ(仮)
"""
st.markdown(markdown)

sys.path.append('../backend')
sys.path.append('../backend/poisson-image-editing/')
import kao
import change_background

########機能1 顔交換プログラム#########

anime_directory = "../images/characters"

image_list = os.listdir(anime_directory)

selected_anime_image = st.selectbox("アニメ画像を選択", image_list)

anime_img_path = os.path.join(anime_directory, selected_anime_image)

if selected_anime_image is not None:
    image = Image.open(anime_img_path)
    st.image(image, caption="選択した画像", use_column_width=True)

human_file = st.file_uploader("人間の画像(.png，.jpeg)", type=['png', "jpeg"])
human_img_path = ''

if human_file != None:
    human_img = Image.open(io.BytesIO(human_file.read()))
    human_img.save((human_image_path := '../images/human/' + human_file.name))
    print(anime_img_path)

# アニメ画像と人間画像のパスを取得
#human_img_path = '' # ../images/human/何たら
#anime_img_path = '' # ../images/characters/何たら
#  ボタンを押したら
ret = st.button("画像を生成!")
if ret:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    output_path = kao.face_exchange(human_image_path, anime_img_path, predictor) # ../images/output/何たら
    out = Image.open(output_path)
    st.image(out)


###################################



########機能2 背景交換プログラム#########

#顔交換済みのコスプレ画像をアップロード
uploaded_completed_file = st.file_uploader("ファイルアップロードpngかjpg", type=["png", "jpg"])

if uploaded_completed_file != None:
    img1 = Image.open(io.BytesIO(uploaded_completed_file.read()))

    markdown = """
    ### 背景画像を選択してください
    """
    st.markdown(markdown)


    ###背景画像を表示して選ぶ機能###

    image_directory = "../images/backgrounds"

    image_list = os.listdir(image_directory)

    selected_image = st.selectbox("画像を選択", image_list)

    image_path = os.path.join(image_directory, selected_image)
    
    img2=image_path

    ############################


    # 背景を削除したい画像(img1)を一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        img1.save(temp_file.name)
        temp_file_path = temp_file.name

    img1=temp_file_path

    #背景交換実行
    change_background.change_background(img1,img2)
    
    #一時ファイルを削除
    os.remove(temp_file_path)

    #完成画像img3をimagesのuploadedに保存(おそらくrunを止めた時点の画像が保存される)
    img3=Image.open("../images/uploaded/composite_image.png")
    img3_array=np.array(img3)

    #完成した画像の表示
    st.image(img3_array, caption="完成画像", use_column_width=True)

###################################
