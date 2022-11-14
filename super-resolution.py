from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
from convMixer import get_Convmixer_SR


val_path_DIV = 'image SR/DIV2K_valid_HR/'
val_path_BSD100 = 'image SR/BSDS100/'
val_path_general100 = 'image SR/General100/'


img = cv2.imread(val_path_DIV+file)
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape
if h%2 > 0:
    h = h-1
if h%4 > 0:
    h = h - 2
if w%2 > 0:
    w = w - 1
if w%4 > 0:
    w = w - 2

img = img[0:h, 0:w, :]
model = get_Convmixer_SR(int(h/4),int(w/4) )
model.load_weights('DIV2k.h5')
img = cv2.resize(img1, (int(w/4) , int(h/4)), interpolation=cv2.INTER_CUBIC)

img = img[tf.newaxis,...] 
pred = model(img)
pred = np.array(pred[0,:,:,:])
pred[pred < 0] = 0
pred[pred > 1.0] = 1.0

out = np.array(pred*255, dtype= np.uint8)
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
pred = pred[0,...]


plt.figure()
plt.imshow(pred)
plt.show()
