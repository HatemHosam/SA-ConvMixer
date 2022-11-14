from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

#activation block
def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    # pixelshuffle downsampling and Conv2D for feature reshaping
    x = tf.nn.space_to_depth(x, patch_size)
    x = layers.Conv2D(filters, kernel_size=1, strides=1)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = activation_block(x)
    x = layers.Add()([x, x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_SA_Convmixer_ADE(
    image_size=320, filters=512, depth=16, kernel_size=5, patch_size=5, out_channels=150):

    inputs = keras.Input((image_size, image_size, 3))
    x = tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance= [0.229**2, 0.224**2, 0.225**2])(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)
    
    concat_layers = []
    # ConvMixer blocks.
    for d in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)
        if d >= 7:
            concat_layers.append(x)
    # segmentation block.
    x = layers.Concatenate()(concat_layers)
    x = layers.Conv2D(num_classes * (patch_size**2), kernel_size=1)(x)
    x = activation_block(x)
    outputs = tf.nn.depth_to_space(x, patch_size)
    outputs = tf.keras.activations.softmax(outputs)

    return keras.Model(inputs, outputs)

    return keras.Model(inputs, outputs)
    
def get_SA_Convmixer(
    image_size=320, filters=512, depth=12, kernel_size=5, patch_size=5, out_channels=21, dataset = 'VOC'):

    inputs = keras.Input((image_size, image_size, 3))
    
    if dataset = 'VOC' or dataset = 'Cityscapes-seg':
        x = tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance= [0.229**2, 0.224**2, 0.225**2])(inputs)
    else:
        x = tf.keras.layers.Rescaling(scale=1./255 )(inputs)
    
    if SR :
        # if the task is super-resolution no downsampling is applied to the input
        x = layers.Conv2D(filters, kernel_size=1, strides=1)(x)
        x = activation_block(x)
    else:
        # Extract patch embeddings.
        x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # feature upsampling block.
    x = layers.Conv2D(num_classes * (patch_size**2), kernel_size=1)(x)
    x = activation_block(x)
    outputs = tf.nn.depth_to_space(x, patch_size)
    # if classification, add softmax activation
    if if dataset = 'VOC' or dataset = 'Cityscapes-seg' or dataset = 'ADE20K':
        outputs = tf.keras.activations.softmax(outputs)

    return keras.Model(inputs, outputs)
    
def get_conv_mixer_SR(
    image_size_h=120, image_size_w=120, filters=256, depth=12, kernel_size=5, patch_size=1, num_channel=3, upscale= 4):

    inputs = keras.Input((image_size_h, image_size_w, 3))
    x = tf.keras.layers.Rescaling(scale=1./255 )(inputs)
    # no patch embeddings.
    x = layers.Conv2D(filters, kernel_size=1, strides=1)(x)
    x = activation_block(x)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.Conv2D(num_channel * (patch_size**2)* (upscale**2), kernel_size=1)(x)
    x = activation_block(x)
    outputs = tf.nn.depth_to_space(x, patch_size * upscale)

    return keras.Model(inputs, outputs)