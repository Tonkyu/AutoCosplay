# streamlit run app.pyというコマンドをターミナルで実行して表示されるURLに以下を構築
import streamlit as st
import cv2
import numpy as np
from PIL import Image

img = cv2.imread("data/chicky_512.png")
st.image(img)
uploaded_file = st.file_uploader("ファイルアップロードpngかjpg", type=["png", "jpg"])
if uploaded_file != None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    #st.write(img_array.shape)
    st.image(img_array, caption="あなたがアップロードした画像", use_column_width=True)
    #st.write(type(uploaded_file))
    uploaded_file_path = "../images/uploaded." + image.format.lower()
    print(uploaded_file_path)
    cv2.imwrite(uploaded_file_path, img_array)
