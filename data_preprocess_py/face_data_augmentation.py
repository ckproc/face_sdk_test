import cv2
import glob
import numpy as np
import scipy.ndimage.filters as fi
import random
import os
from random import sample, randint
from scipy import misc

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def add_block(image):
    height, width, _ = image.shape
    face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
    face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]
    #print image.shape
    seed = np.random.randint(0,4)
    mask = np.ones((height, width), np.uint8) * 255
    #print mask.shape
    #seed = 1
    if seed == 0:          # add block on left
        cv2.rectangle(mask, (0,0), (int(0.2*width), height), color=(0,0,0), thickness=-1)
    elif seed == 1:         # add block on top
        cv2.rectangle(mask, (0,0), (width, int(0.2*height)), color=(0,0,0), thickness=-1)
    elif seed == 2:         # add block on right
        cv2.rectangle(mask, (int(0.8*width),0), (width, height), color=(0,0,0), thickness=-1)
    elif seed == 3:         # add block on bottom
        cv2.rectangle(mask, (0, int(0.8*height)), (width, height), color=(0,0,0), thickness=-1)
    
    masked_data = cv2.bitwise_and(image,image, mask=mask)
    return masked_data

def adjust_local_illumination(image):
    width, height, _ = image.shape
    masks = []
    range_x = (int(width*0.2), int(width*0.8))
    range_y = (int(height*0.2), int(height*0.8))
    cx, cy = randint(range_x[0], range_x[1]), randint(range_y[0], range_y[1])
    for i in range(1,30):
        mask = np.zeros(image.shape[:2], np.uint8)
        if i == 1:
            cv2.circle(mask, (cx, cy), 3*i, 255, -1)
        else:
            cv2.circle(mask, (cx, cy), 3*i, 255, -1)
        masks.append(mask)
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        masked_img = adjust_gamma(masked_img, gamma=1.0+0.05*(30-i))
        #masked_img =adjust_gamma(masked_img, gamma=1.0-(30-i)/60.)
        if i == 1:
            final = masked_img
        else:
            final = final + cv2.bitwise_and(masked_img, masked_img, mask=(255-masks[-2]))
    unmasked_img = cv2.bitwise_and(image, image, mask=(255-mask))
    final = cv2.bitwise_or(final, unmasked_img)
    return final

gamma_list = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]        # change the value here to get different result
NUM_PER_IMG = 15

data_source = "./finn_crop_clean"
faceid=os.listdir(data_source)
p=0
for sub in faceid:
 subdir="./finn_crop_clean_resize/"+sub
 if(not os.path.exists(subdir)):
  os.mkdir(subdir)
 p+=1
 namedir =os.listdir(data_source+"/"+sub)
 for filename in namedir:
  img_nm = data_source+"/"+sub+"/"+filename
  img = misc.imread(img_nm)
  scaled = misc.imresize(img, (160, 160), interp='bilinear')
  w = subdir+"/"+filename
  misc.imsave(w,scaled)
  #print(img_nm)
  #original = cv2.imread(img_nm)
  #cv2.resize(original,original,)
  #if original is None:
  # print(img_nm)
  # continue
  #for i in range(NUM_PER_IMG):
  #  gamma = sample(gamma_list, 1)[-1]
    #print gamma
  #  if random.random() > 0.5:
  #      adjusted = adjust_gamma(original, gamma=gamma)
  #  else:
  #      adjusted = adjust_local_illumination(original)
  #  masked_data = add_block(adjusted)
  #  write_path = data_source+"/"+sub+"/"+os.path.splitext(os.path.split(img_nm)[1])[0]+"_"+str(i+1000)+".jpg"
    #print write_path
  #cv2.imwrite(write_path,masked_data)
 print p

