# streamlit run app.py
import streamlit as st
import cv2

img = cv2.imread("data/chicky_512.png")
st.image(img)