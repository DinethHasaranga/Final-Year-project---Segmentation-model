from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2

def convolution_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# no of filters means no of output channels
# x -> output of conv layer or the skip connection
# p -> output of pooling layer
def encoder_block(inputs, num_filters):
    x = convolution_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

# The step size used in the convolution process is set by the strides parameter. Here, a stride of 2 indicates that the
# input will be upsampled by a factor of two as the transposed convolution operation moves by 2 pixels at a time.

# The required amount of padding is added in "same padding" so that the output and input sizes are the same.
def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = convolution_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    # s, p -> x, p where x is skip connection and p is output of pooling layer
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # print(s1.shape, s2.shape, s3.shape, s4.shape)
    # print(p1.shape, p2.shape, p3.shape, p4.shape)

    b1 = convolution_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_unet(input_shape)
    model.summary()