# -*- coding: utf-8 -*-
import re, os, os.path,shutil
import time
import glob
import random
import cv2

#path="./source_best_oy_filter"
path="/workspace2/yyl/dataset/cleaned_asia_faces"
topath="./cleaned_asia_faces"
faceid=os.listdir(topath)
print len(faceid)
p=0
for sub in faceid:
        #p+=1
        #if(p <4485):
        #continue
        namedir=os.listdir(topath+"/"+sub)
        #if(len(namedir)==0):
        # print sub
        # p+=1
        #continue
        file_dir = topath+"/"+sub
        #print file_dir
        #if not os.path.exists(file_dir):
        # os.mkdir(file_dir)
        for filename in namedir:
         #filedir = path+"/"+sub+"/"+filename
         todir = file_dir+"/"+filename
         #print filedir
         img=cv2.imread(todir)
         if(img is None):
          print todir
          p+=1
          os.remove(todir)
         #continue
         #if not (img.shape[0]==160 and img.shape[1]==160):
          #img=cv2.resize(img,(160,160))
          #continue
         #cv2.imwrite(todir,img)
         #shutil.copyfile(filedir,todir)
        
print p

            
