# app.py
# streamlit run app.pyというコマンドをターミナルで実行して表示されるURLに以下を構築
import streamlit as st
import cv2
import numpy as np
import sys
import io
import os
import tempfile
from PIL import Image

markdown = """
# プロジェクト演習E班 成果物
##  どこでもコスプレアプリ(仮)
"""
st.markdown(markdown)

#backendにあるpythonファイル(create_cosplay_image.py,change_background.py)を動かすためにファイルパスを通す
sys.path.append("../backend")


#sys.pathを用いて動かしたいファイルをインポート
#import create_cosplay_image.py
import change_background


########機能1 顔交換プログラム#########

###################################



########機能2 背景交換プログラム#########

#顔交換済みのコスプレ画像をアップロード
uploaded_completed_file = st.file_uploader("ファイルアップロードpngかjpg", type=["png", "jpg"])

if uploaded_completed_file != None:
    #st.file_uploader でアップロードされた PNG ファイルを PIL の Image オブジェクトに変換 (PILライブラリを使用したいため)
    img1 = Image.open(io.BytesIO(uploaded_completed_file.read()))

    markdown = """
    ### 背景画像を選択してください
    """
    st.markdown(markdown)
    

    ###背景画像を表示して選ぶ機能###

    # 画像を保存しているディレクトリのパス
    image_directory = "../images/backgrounds"

    # 画像一覧を取得
    image_list = os.listdir(image_directory)

    # 画像選択用のセレクトボックスを表示
    selected_image = st.selectbox("画像を選択", image_list)

    # 選択した画像のファイルパスを取得
    image_path = os.path.join(image_directory, selected_image)

    # 画像を表示
    if selected_image is not None:
        image = Image.open(image_path)
        st.image(image, caption="選択した画像", use_column_width=True)

    #選んだ背景画像をimg2に渡す
    img2=image_path

    ############################


    # 背景を削除したい画像(img1)を一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        img1.save(temp_file.name)
        temp_file_path = temp_file.name

    #img1はこのままだとPngImageFileオブジェクトであり、open()関数で直接扱えないので一時ファイルのファイルパスを渡す。
    img1=temp_file_path

    #背景交換実行
    change_background.change_background(img1,img2)
    
    #一時ファイルを削除
    os.remove(temp_file_path)

    #完成画像img3をimagesのuploadedに保存
    img3=Image.open("../images/uploaded/composite_image.png")
    img3_array=np.array(img3)

    #完成した画像の表示
    st.image(img3_array, caption="完成画像", use_column_width=True)

###################################
