import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import kao5
import dlib
from PIL import Image
import numpy as np

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
st.title("Streamlit App Test")
st.write("Hello world")
if "cnt" not in st.session_state:
    st.session_state.cnt = 0
    print(st.session_state.cnt)
if "process_set" not in st.session_state:
    st.session_state.process_set = 0
class VideoProcessor:
    def recv(self, frame):
        img, st.session_state.process_set = kao5.face_exchange(frame.to_ndarray(), "lennon.jpeg", predictor)
        st.session_state.cnt += 1
        img = av.VideoFrame.from_ndarray(img, format="gray")
        return img
        if st.session_state.cnt == 0:

            #if "process_set" not in st.session_state:
            img, st.session_state.process_set = kao5.face_exchange(frame.to_ndarray(), "lennon.jpeg", predictor)
            st.session_state.cnt += 1
            img = av.VideoFrame.from_ndarray(img)
            return img
        else:
            img = st.session_state.process_set.base_process(Image.fromarray(frame.to_ndarray()))
            img = av.VideoFrame.from_ndarray(np.array(img))
            return img
"""
def video_frame_callback(frame):
    img = frame.to_ndarray()
    if st.session_state.cnt == 0:
        img, st.session_state.process_set = kao5.face_exchange(img, "lennon.jpeg", predictor)
        st.session_state.cnt += 1
        return av.VideoFrame.from_ndarray(img)
    else:
        img = st.session_state.process_set.base_process(Image.fromarray(img))
        return av.VideoFrame.from_ndarray(img)
"""
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
