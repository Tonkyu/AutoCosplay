import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import kao5

st.title("Streamlit App Test")
st.write("Hello world")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format = "bgr24")
        return img

webrtc_streamer(key="example")