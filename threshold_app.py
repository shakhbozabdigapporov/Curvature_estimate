import streamlit as st
import cv2
import numpy as np
from PIL import Image

def thresholding(img, sobel_thresh_min, sobel_thresh_max, color_thresh_min, color_thresh_max):
    # Convert to HLS color space and separate the L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x to detect gradient in the x direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_thresh_min) & (scaled_sobel <= sobel_thresh_max)] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh_min) & (s_channel <= color_thresh_max)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

# Streamlit app
st.title('Dynamic Threshold Tuning for Image Thresholding')
st.write('Use the sliders to adjust the threshold values and see the changes in real-time.')

# Load the image
# uploaded_file = st.file_uploader("Choose an image...", type="png")

# if uploaded_file is not None:
    # Convert the file to an opencv image
    # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # img = cv2.imdecode(file_bytes, 1)
img = cv2.imread('img.png')

# Display original image
st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

# Sobel threshold sliders
st.sidebar.header('Sobel Thresholds')
sobel_thresh_min = st.sidebar.slider('Sobel Min Threshold', 0, 255, 20)
sobel_thresh_max = st.sidebar.slider('Sobel Max Threshold', 0, 255, 100)

# Color threshold sliders
st.sidebar.header('Color Thresholds')
color_thresh_min = st.sidebar.slider('Color Min Threshold', 0, 255, 170)
color_thresh_max = st.sidebar.slider('Color Max Threshold', 0, 255, 255)

# Apply thresholding
binary_img = thresholding(img, sobel_thresh_min, sobel_thresh_max, color_thresh_min, color_thresh_max)

# Display thresholded image
st.image(binary_img * 255, caption='Thresholded Image', use_column_width=True, clamp=True)
