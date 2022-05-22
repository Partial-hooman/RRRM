import cv2
import numpy as np
import streamlit as st
from  PIL import Image, ImageEnhance 
import tempfile
import subprocess as sp

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
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())
   
      

    vf = cv2.VideoCapture(tfile.name)
    #returnus, frames = vf.read()
    #result = cv2.VideoWriter('output.mp4', 
                         #cv2.VideoWriter_fourcc(*'x264'),
                         #20, frames.shape[:2])
    
    #ffmpeg = 'ffmpeg'
    #dimension = '{}x{}'.format(frames.shape[0], frames.shape[1])
    #f_format = 'bgr24' # remember OpenCV uses bgr format
    #fps = str(vf.get(cv2.CAP_PROP_FPS))
    
    
    #command = [ffmpeg,
            #'-y',
            #'-f', 'rawvideo',
            #'-vcodec','rawvideo',
            #'-s', dimension,
            #'-pix_fmt', 'bgr24',
            #'-r', fps,
            #'-i', '-',
            #'-an',
            #'-vcodec', 'mpeg4',
            #'-b:v', '5000k',
            #'output_file_name.mp4']

    #proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    stframe = st.empty()
      
    
    
    
    
    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        proc_frame =  conv2manga(frame)
        dst2 = cv2.detailEnhance(proc_frame, sigma_s=10, sigma_r=0.15)
        #result.write(dst2)
        #proc.stdin.write(dst2.tobytes())
        #if frame is None:
           #break
        frameBytes = cv2.imencode('.png', dst2)[1].tobytes()
        with st.empty():
             stframe.image(dst2)
        #stframe.video(dst2)
    

    #result.release()
    #vf.release()
    #proc.stdin.close()
    #proc.stderr.close()
    #proc.wait()
    
    #video_file = open('output_file_name.mp4', 'rb')
    #video_bytes = video_file.read()

    #st.video(video_bytes)
    
