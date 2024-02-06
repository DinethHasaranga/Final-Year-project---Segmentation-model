import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from unet_model import build_unet
from metrics import dice_loss, dice_coef
import albumentations as A

# Dimensions of input image
H = 256
W = 256

# Creating a directory using a path. It will first check if path exist if not will create
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Loading the images and respective masks.
# train_x : training images
# train_y : training masks
# glob function is used to extract all from the given path and extension
# sorted function is used to make sure images and respective masks are loaded in the same order.
def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

# Data augmentation
def augment_data(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

    ])

    augmented = transform(image=image, mask=mask)
    aug_image, aug_mask = augmented["image"], augmented["mask"]

    return aug_image, aug_mask

def tf_parse(x, y, save_augmented=False, epoch=None, counter=None):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        x, y = augment_data(x, y)

        # if save_augmented and epoch is not None and epoch == 0:
        #     # Save the augmented images and masks with unique filenames
        #     create_dir("augmented_images")
        #     filename = f"augmented_images/images/{counter.numpy()}.png"
        #     cv2.imwrite(filename, (x * 255).astype(np.uint8))
        #     filename = f"augmented_images/masks/{counter.numpy()}.png"
        #     cv2.imwrite(filename, (y[:, :, 0] * 255).astype(np.uint8))

        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2, save_augmented=False, epoch=None):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    # Counter variable
    counter = tf.Variable(0, dtype=tf.int32)

    def update_counter(x, y):
        # Increment the counter for each batch
        counter.assign_add(1)
        return x, y

    # Apply the update_counter function to the dataset
    dataset = dataset.map(update_counter)

    dataset = dataset.map(lambda x, y: tf_parse(x, y, save_augmented, epoch, counter))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":
    # Seeding the environment
    np.random.seed(42)
    tf.random.set_seed(42)

    # Creating a directory to store files
    create_dir("files")

    # Hyperparameters
    # Batch Size -> Number of samples processed in a single iteration during training.
    # Learning rate -> Step size at each iteration
    batch_size = 4
    lr = 1e-4
    num_epochs = 5
    model_path = os.path.join("files", "unet_model_with_augmentation.h5")
    csv_path = os.path.join("files", "unet_log_with_augmentation.csv")

    # Loading the Dataset
    dataset_path = "../data_folder"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    # Save augmented images and masks for the first epoch
    train_dataset_first_epoch = tf_dataset(train_x, train_y, batch=batch_size, save_augmented=True, epoch=0)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # Compiling the Model
    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset_first_epoch,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
