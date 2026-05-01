import os
import shutil
import random

# Paths
base_dir = os.getcwd()
source_dataset = os.path.join(base_dir, "dataset")
classes = ["bird", "drone"]

# Destination paths
dest_root = os.path.join(base_dir, "classification_data")
splits = ["train", "val", "test"]

# Create directories
for s in splits:
    for c in classes:
        os.makedirs(os.path.join(dest_root, s, c), exist_ok=True)

# Split parameters
train_perc = 0.8
val_perc = 0.1
# Test will be the remaining 0.1

for c in classes:
    class_path = os.path.join(source_dataset, c)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * train_perc)
    n_val = int(n * val_perc)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    print(f"Class {c}: Total={n}, Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    
    # Copy images
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(dest_root, "train", c, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(dest_root, "val", c, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(dest_root, "test", c, img))

print("Data splitting complete.")
