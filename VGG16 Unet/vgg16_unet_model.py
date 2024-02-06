from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import tensorflow as tf

print("TF Version: ", tf.__version__)

def convolution_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = convolution_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    # Input layer
    inputs = Input(input_shape)

    # Using pretrained vgg16 model
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encorder block
    s1 = vgg16.get_layer("block1_conv2").output         # (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         # (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         # (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output         # (64 x 64)

    # The bridging block to connect the encorder and decorder
    b1 = vgg16.get_layer("block5_conv3").output         # (32 x 32)

    # Decorder block
    d1 = decoder_block(b1, s4, 512)                     # (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     # (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     # (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      # (512 x 512)

    # Ouput layer
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_vgg16_unet(input_shape)
    model.summary()