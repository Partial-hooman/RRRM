import cv2
import numpy as np
import streamlit as st
from  PIL import Image, ImageEnhance 


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
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)




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
    auto_result, alpha, beta = automatic_brightness_and_contrast(bl2)
    dst = cv2.detailEnhance(auto_result, sigma_s=10, sigma_r=0.15)

    return dst




st.title('Conv2manga.IO')

image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    image = Image.open(image_file)
    converted_img = np.array(image.convert('BGR'))
    proc_img = conv2manga(converted_img)
    st.image(proc_img, width=300)

