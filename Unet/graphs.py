import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
log_data = pd.read_csv("files/unet_log.csv")
plt.figure(figsize=(12, 12))

# Plotting Training and Validation Loss

plt.plot(log_data['epoch'], log_data['loss'], label='Training Loss', marker='o', color='b')
plt.plot(log_data['epoch'], log_data['val_loss'], label='Validation Loss', marker='o', color='r')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Dice Coef (Accuracy)

plt.figure(figsize=(12, 12))
plt.plot(log_data['epoch'], log_data['dice_coef'], label='Training Dice Coef', marker='o', color='g')
plt.plot(log_data['epoch'], log_data['val_dice_coef'], label='Validation Dice Coef', marker='o', color='purple')
plt.title('Training and Validation Dice Coef')
plt.xlabel('Epoch')
plt.ylabel('Dice Coef')
plt.legend()
plt.show()

plt.figure(figsize=(12, 12))

# Plotting Learning Rate
plt.plot(log_data['epoch'], log_data['lr'], label='Learning Rate', marker='o', color='orange')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.legend()
plt.show()