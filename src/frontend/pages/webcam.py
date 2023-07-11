# -*- coding: utf-8 -*-

import streamlit as st
import cv2
from PIL import Image #, ImageOps
import kao5
import numpy as np
import dlib
import sys

if "cnt" not in st.session_state:
    st.session_state.cnt = 0
st.session_state.cnt = st.button("stop")
if st.session_state.cnt == True:
    cv2.waitKey(0)

cap = cv2.VideoCapture(0)

image_loc = st.empty()
cap = cv2.VideoCapture(0)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
ret, src = cap.read()
if not ret:
    print("Can not open camera")
    sys.exit()
    
base_array = np.array(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
output_array, process_set = kao5.face_exchange(base_array, "lennon.jpeg", predictor)
img = Image.fromarray(output_array)
#img = ImageOps.mirror(img)
image_loc.image(img)

while True:
    ret, src = cap.read()
    if not ret:
        break

    img = process_set.base_process(Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)))
    #img = ImageOps.mirror(img)
    image_loc.image(img)