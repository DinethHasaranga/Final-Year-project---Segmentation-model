import tensorflow as tf

# To prevent division by zero, a small constant number (1e-15) is added to the
# denominator and numerator in the dice coefficient computation.
smooth = 1e-15

# Calculates the Dice coefficient between the true labels (y_true) and the predicted labels (y_pred)
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Loss is used as the objective function to be minimized during the training of a neural
# network for segmentation tasks
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)