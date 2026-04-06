import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_dir = os.getcwd()
test_dir = os.path.join(base_dir, "classification_data", "test")
IMG_SIZE = (224, 224)

# Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

def evaluate_model(model_path, name):
    print(f"\nEvaluating {name}...")
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    preds = model.predict(test_generator)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Report
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bird', 'Drone'], yticklabels=['Bird', 'Drone'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'cm_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Evaluate both models if they exist
if os.path.exists('custom_cnn_model.keras'):
    evaluate_model('custom_cnn_model.keras', 'Custom CNN')

if os.path.exists('transfer_learning_model.keras'):
    evaluate_model('transfer_learning_model.keras', 'Transfer Learning')
