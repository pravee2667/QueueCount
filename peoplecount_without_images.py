# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:57:18 2019

@author: P0142221
"""

from keras_retinanet import models
from keras_retinanet.utils.image import  preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.colors import label_color
import cv2,os,time,sys
import numpy as np
import pushservice as ps
import DBconnect as db
import psutil
import pandas as pd
import tensorflow as tf
from keras import backend as K

#If you would like TensorFlow to automatically choose an existing and 
#supported device to run the operations in case the specified one doesn't 
#exist, you can set allow_soft_placement 
#to True in the configuration option when creating the session

num_of_cores=4
config=tf.ConfigProto(intra_op_parallelism_threads=num_of_cores,
                      inter_op_parallelism_threads=num_of_cores,
                      allow_soft_placement=True,
                      device_count={'CPU':num_of_cores})

session=tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"]="num_of_cores"
os.environ["KMP_BLOCKTIME"]="30"
os.environ["KMP_SETTINGS"]="1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


def execute(inp,queu=None):
    if queu is None:
        queu=[]
    try:
        video_inp = inp[1]
        video=inp[2]
        queue=inp[3]
        print(video_inp)
        model_path = 'model1.h5'
        
        print("#################")
        model = models.load_model(model_path, backbone_name='resnet50')
        #model=models.convert_model(model)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30,30)
        fontScale              = 1
        fontColor1             = (255,0,0)
        fontColor2             = (0,255,0)
        lineType               = 2
        video_reader = cv2.VideoCapture(video_inp)
        if video_reader is None or not video_reader.isOpened():
            ps.pusherrrorininput("Q2")
        else:
            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            #nb_fps= int(video_reader.get(cv2.CAP_PROP_FPS))
            start = 0
            FRS = 150
            time_,people_=[],[]
            i=video
            k=queue
            print("Queueu is {}".format(queue))
            processid=os.getpid()
            for j in range(0,nb_frames,FRS):
                video_reader.set(1, j)
                time_sec=int(video_reader.get(0))
                print("Current time in the Video {}".format((time_sec/1000)))
                print("Memory allocated is  {}".format(psutil.Process(processid).memory_info()[0]))
                res, image = video_reader.read()
                draw = image.copy()
                draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                if i=="V1": 
                    print("QUEUUE LENTH....{}".format(i))
                    crop_img_g=draw[200:600,100:272].copy()   
                elif i=="V2":
                    print("QUEUUE LENTH....{}".format(i))
                    crop_img_g=draw[200:600,100:300].copy()   
                elif i=="V3": 
                    if k=="Q3":
                        crop_img_g=draw[100:600,200:320].copy()  
                    elif k=="Q4":
                        crop_img_g=draw[100:600,400:550].copy()  
                    else:
                        crop_img_g=draw[100:600,200:320].copy()
                elif i=="V4":
                    print("QUEUUE LENTH....{}".format(i))
                    if k=="Q5":
                        crop_img_g=draw[100:600,100:270].copy()  
                    elif k=="Q6":
                        crop_img_g=draw[100:600,250:350].copy()  
                    else:
                        crop_img_g=draw[100:600,80:250].copy()  
                    #crop_img_g=draw[100:600,80:250].copy()   
                elif i=="V6":
                    print("QUEUUE LENTH....{}".format(i))
                    crop_img_g=draw[100:600,150:330].copy() 
                else:
                    print("QUEUUE LENTH....{}".format(i))
                    crop_img_g=draw[100:600,150:330].copy() 
                    
              
                crop_img_gg = preprocess_image(crop_img_g)
                crop_img_gg, scale = resize_image(crop_img_gg)
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(crop_img_gg, axis=0))
                count1 = sum(( scores[0] > 0.5) * (labels[0] == 0))
                time_.append(time_sec)
                people_.append(count1)
                db.connection(count1,queue)
                db.mysql_connection(count1,queue)
                print(count1)
                queu.append(count1)
    
                ps.pushser(count1,queue)
                start=start+1
              
               
                time.sleep(1)
                boxes /= scale
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    if score < 0.5:
                        break
                    color = label_color(label)
                    b = box.astype(int)
                    if label == 0:
                        draw_box(crop_img_g, b, color=color)
                if count1 > 5:
                    fontColor = fontColor1
                else:
                    fontColor = fontColor2
                cv2.putText(crop_img_g,str(count1), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
                cv2.waitKey(1)
                time.sleep(1)
                cv2.destroyAllWindows()
    except KeyboardInterrupt:
            print("Queue")
    except IndexError as er:
            print("The Exception is {}".format(er))
            ps.pushinputerror("Q3")
            
            
        
    
                

if __name__ == "__main__":
    execute(sys.argv)
 
    
