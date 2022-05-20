import cv2
import numpy as np
import streamlit as st
from  PIL import Image, ImageEnhance 



def conv2manga(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    edges  = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)




    n = 8

    for i in range(n):
        im_gray[(im_gray >= i*255/n) & (im_gray < (i+1)*255/n)] = i*255/(n-1)

    cartoon = cv2.bitwise_and(im_gray, im_gray, mask=edges)


    ret, thresh1 = cv2.threshold(cartoon, 70, 255, cv2.THRESH_BINARY)
    





    blend = cv2.addWeighted(thresh1, 0.5, cartoon, 0.8, 0.0)

    bl2 = cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
    
    return bl2



def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
   

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)





st.title('Conv2manga.IO')

image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    image = Image.open(image_file)
    converted_img = np.array(image)
    proc_img = conv2manga(converted_img)
    auto_result, alpha, beta = automatic_brightness_and_contrast(proc_img)
    dst = cv2.detailEnhance(auto_result, sigma_s=10, sigma_r=0.15)

    st.image(dst, width=none)

