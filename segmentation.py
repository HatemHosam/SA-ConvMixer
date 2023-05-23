from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os
import cv2
from colormaps import VOC_cm, ADE_cm, Cityscapes_cm
from convMixer import get_SA_Convmixer_ADE, get_SA_Convmixer

dataset = 'VOC'     # can choose also 'ADE20K' or 'Cityscapes-Seg'

if dataset == 'ADE20K':
    model = get_SA_ConvMixer_ADE20k(
                image_size=320, filters=512, depth=16, kernel_size=5, patch_size=5, num_classes=150)
    model.load_weights('ADE20K.h5')
    cmap = ADE_clr

elif dataset == 'VOC':
    model = get_SA_ConvMixer(
                image_size_h=320, image_size_w=320, filters=512, depth=12, kernel_size=5, patch_size=5, SR = False, num_classes=21, dataset='VOC')
    model.load_weights('VOC.h5')
    cmap = VOC_clr

elif dataset == 'Cityscapes-Seg':
    model = get_SA_ConvMixer(
                image_size_h=256, image_size_w=512, filters=512, depth=12, kernel_size=5, patch_size=5, SR = False, num_classes=21, dataset='VOC')
    model.load_weights('Cityscapes_seg.h5')
    cmap = Cityscapes_clr
    
img = cv2.imread('img.jpg')
img = cv2.resize(img, (320,320))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
img = img[tf.newaxis,...]

pred = model.predict(img)
pred = pred[0,...]
pred_mask = np.uint8(np.argmax(pred, axis=-1))

seg_color = np.zeros((320,320,3), dtype= np.uint8)
for i in range(len(cmap)):
    seg_color[pred_mask==i] = cmap[i]

plt.figure()
plt.imshow(seg_color)
plt.show()
