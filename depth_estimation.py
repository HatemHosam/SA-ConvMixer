from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
from convMixer import get_SA_Convmixer


dataset = 'NYUV2'     # can choose also'Cityscapes-depth'


elif dataset == 'NYUV2':
    model = get_SA_ConvMixer(
                image_size_h=480, image_size_w=640, filters=256, depth=12, kernel_size=5, patch_size=5, dataset = 'NYUV2')
    model.load_weights('NYU-depthV2.h5')

elif dataset == 'Cityscapes-depth':
    model = get_SA_ConvMixer(
                image_size_h=512, image_size_w=1024, filters=256, depth=12, kernel_size=5, patch_size=8, dataset = 'Cityscapes-depth')
    model.load_weights('Cityscapes-depth.h5')
    
img = cv2.imread('img.jpg')
img = cv2.resize(img, (1024,512))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
img = img[tf.newaxis,...]

pred = model.predict(img)
pred = pred[0,...]


plt.figure()
plt.imshow(seg_color, cmap= 'magma')
plt.show()
