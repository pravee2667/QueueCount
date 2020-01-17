# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:16:50 2019

@author: P0142221
"""

from pyfcm import FCMNotification
import DBconnect as db
push_service = FCMNotification(api_key="AAAAq1-5G7Q:APA91bHubu1C2sQUBKjkBhEVhOTUSCck7DVLKS-4YRcoSmg93J7Og83wzZiCtLV7ALMQkQ32SvxC4i1fJd_y5scBhaoLSGb7tlMzOqutBF0i0LQb1EkW4LuEzCxIAm-CNwT_Thd1LPhv")
registration_id="fjT6n03BK_w:APA91bHOsF6FtsuJ2uOWoSAA9nDIea8erGRM8jqg2Nc0HncX29E4erycVvMLgoWE_i_eUmGU0wpoENjsC5-RKgnM937lb-PsLrFtQ3qmXy5uUVhNpUhxBcresq-JCgp1tJXjzq8EzJ_L"
a=dict()
def pushser(count,que_num):
    a[que_num]=count
    push_service.notify_single_device(registration_id=registration_id, message_body=str(a))
    if count>2:
        sample_list=db.fcm_connection()
        for i in sample_list:
            push_service.notify_single_device(registration_id=i, message_body="Threshold Limit has reached to {} for {}".format(count,que_num))

def pushinputerror(que_num):
    sample_list=db.fcm_connection()
    for i in sample_list:
        push_service.notify_single_device(registration_id=i,message_body="Queue Number {} is having issues.Please look into it.".format(que_num))
    
def pusherrrorininput(que_num):
    sample_list=db.fcm_connection()
    for i in sample_list:
        push_service.notify_single_device(registration_id=i,message_body="Error Opening in input file {}.Please look into it.".format(que_num))
    
        