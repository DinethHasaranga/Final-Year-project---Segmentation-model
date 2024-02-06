import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
log_data = pd.read_csv("files/vgg16_log.csv")
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

# Load the scoring CSV file
scoring_data = pd.read_csv("files/score.csv")

# Extract data for each metric
images = scoring_data['Image']
f1_scores = scoring_data['F1']
jaccard_scores = scoring_data['Jaccard']
recall_scores = scoring_data['Recall']
precision_scores = scoring_data['Precision']

# Plot F1 Scores
plt.figure(figsize=(12, 6))
plt.plot(images, f1_scores, marker='o', label='F1 Score', color='blue')
plt.title('Trend of F1 Scores Across Images')
plt.xlabel('Image')
plt.ylabel('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# Plot Jaccard Scores
plt.figure(figsize=(12, 6))
plt.plot(images, jaccard_scores, marker='o', label='Jaccard Score', color='green')
plt.title('Trend of Jaccard Scores Across Images')
plt.xlabel('Image')
plt.ylabel('Jaccard Score')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# Plot Recall Scores
plt.figure(figsize=(12, 6))
plt.plot(images, recall_scores, marker='o', label='Recall', color='orange')
plt.title('Trend of Recall Scores Across Images')
plt.xlabel('Image')
plt.ylabel('Recall')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# Plot Precision Scores
plt.figure(figsize=(12, 6))
plt.plot(images, precision_scores, marker='o', label='Precision', color='red')
plt.title('Trend of Precision Scores Across Images')
plt.xlabel('Image')
plt.ylabel('Precision')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
