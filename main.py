import cv2
import numpy as np
import streamlit as st
from  PIL import Image, ImageEnhance 
import tempfile
import subprocess as sp
import os




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


    
    
    
    
    
    
    
    
    
    
    
    
f = st.file_uploader("Upload vid", type=['mp4'])

if f is not None:
    
    f = st.file_uploader("Upload file")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    
    def framing(video):#defining a small function named"framing" with a parameter "i" that's supposed to be provided for reading the video
        fr = []#creating an empty list named fr
        fr_pre=[]#creating an empty list named fr_pre
        cap = cv2.VideoCapture(video)#reading the video file
        while (cap.isOpened()):#This command builds a loop to check if the data is still being read from the video
            ret,frame = cap.read()#reading the data tunnel,gives two output where one tells about presence of frames(here it's ret) & the other speaks frame data(here it's frame)
            if frame is None:
                break
            if ret == True:
               cv2.imshow("fbyf",frame)#displaying the frames
               grayed = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#Converting the frames to Grayscale from BGR
               canned = cv2.Canny(grayed,320,320)#For extrating edges we use Canny Edge detection method
               fr.append(frame)#Appending the read frame
               fr_pre.append(canned)#Appending the edge extracted frames
               cv2.imshow("Grayed",grayed)#Displaying the original frames
               cv2.imshow("Canned",canned)#Displaying the edge detected frames
               k = cv2.waitKey(10) & 0XFF#this is an arrangement for displaying the video where the secs for which each frame needs to be displayed in given in the paranthesis

        cap.release()#Here we release the resoures   
        return fr_pre,fr     #returning the frames received after the execution of function
    frames,ogframes = framing(tfile.name)
    diff = []#creatin a list variable
    for i in range(0,len(frames)-1):#defining the range
        print(frames[i],frames[i+1])#checking the frames presence
        diff.append(cv2.absdiff(frames[i],frames[i+1]))#appending the diff between frames to the list variable so we're supposed to get only the difference between frames    
    np.mean(diff)
    mn = np.mean(diff)#This gives mean
    st_d = np.std(diff)#This gives standard deviation
    print(mn,st_d)#we check the mean & standard deviation
    a = 4#Setting a random value we can modify it to any value 
    ts = mn + (a * st_d)#defining the standard threshold value for the project/global threshold value
    print('The threshold==>',ts)
    a_fr = []#Creating an empty list
    for i in range(len(diff)):#Defining the for loop to be looped over all the frames obtained after finding the frames resulted from subtracting
        mn = np.mean(diff[i])#Calculating the mean for each frame
        st_d = np.std(diff[i])#Calculating the standard deviation for each frame
        fr_ts = mn + (4*st_d)#Finding the threshold values for each frame/image
        print(i,fr_ts)
        a_fr.append([i,fr_ts])#Appending the frame number & the threshold values
    imp_fr = []#Creating an empty list
    for i,ac_tr in(a_fr):#Defining the loop on the list obtained from above code
        if ac_tr >= ts:#Comapring the threshold values to the standard threshold/global threshold values
           print(i,ac_tr)
           imp_fr.append([i,ac_tr])#Appending the list with the imp frames based on their index & the values  
    key_fr = []#Creating an empty list
    for i,_ in imp_fr:#Defining the loop over the list obtained from above code
        key_fr.append(ogframes[i])#This extracts the frames based on the index of frames 
        print(diff[i],i)
    st.image(key_fr[0])
