import cv2
import numpy as np
import streamlit as st
from  PIL import Image, ImageEnhance 



def conv2manga(image):
    imgrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges  = cv2.adaptiveThreshold(imgrey, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)


    n = 10

    for i in range(n):
        imgrey[(imgrey >= i*255/n) & (imgrey < (i+1)*255/n)] = i*255/(n-1)

    cartoon = cv2.bitwise_and(imgrey, imgrey, mask=edges)
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
    st.image(proc_img, width=576)

