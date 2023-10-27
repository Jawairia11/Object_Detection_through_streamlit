import streamlit as st
import cv2
from object_detection import perform_object_detection

st.title("YOLOv7 Object Detection with Streamlit")

#this is a streamlit interface where we upload an image and perform object detection
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.image(perform_object_detection(uploaded_image), use_column_width=True)