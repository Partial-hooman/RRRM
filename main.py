import cv2
import numpy as np
import streamlit as st
from  PIL import Image, ImageEnhance 
import tempfile
import subprocess as sp
import os
from streamlit_webrtc import webrtc_streamer
import av



def conv2manga(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    edges  = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    n = 15 # Number of levels of quantization

    indices = np.arange(0,256)   # List of all colors 

    divider = np.linspace(0,255,n+1)[1] # we get a divider

    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors

    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..

    palette = quantiz[color_levels] # Creating the palette

    im_gray = palette[im_gray]  # Applying palette on image

    im_gray = cv2.convertScaleAbs(im_gray) # Converting image back to uint8

    cartoon = cv2.bitwise_and(im_gray, im_gray, mask=edges)


    ret, thresh1 = cv2.threshold(cartoon, 70, 255, cv2.THRESH_BINARY)
    





    blend = cv2.addWeighted(thresh1, 0.5, cartoon, 0.8, 0.0)

    bl2 = cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
    
    return bl2





st.title('Conv2manga.IO')

image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    image = Image.open(image_file)
    converted_img = np.array(image)
    proc_img = conv2manga(converted_img)
    dst = cv2.detailEnhance(proc_img, sigma_s=10, sigma_r=0.15)

    st.image(dst, width=None)


    




class VideoProcessor:
   def recv(self, frame):
       img = frame.to_ndarray(format="bgr24")

       img = conv2manga(img)
       dst2 = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)


       return av.VideoFrame.from_ndarray(dst2, format="bgr24")  
   


webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
    
    
    
    
    
    


    
 
