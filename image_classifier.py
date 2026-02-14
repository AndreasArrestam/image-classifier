import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

def filter_dataset():
    for folder_name in ("Black Sea Sprat", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout"):
        folder_path = os.path.join("Fish_Dataset", folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                file = open(file_path, "rb")
                is_png = b"png" in file.peek(10)
            finally:
                file.close()
                
            #Remove file if it is not a png file
            if not is_png:
                os.remove(file)
                
def generate_dataset(img_size, batch_size):
    train_set, validation_set = keras.utils.image_dataset_from_directory(
        "Fish_Dataset",
        validation_split=0.2, #20% is used for validation
        subset="both",
        seed = 1337,
        image_size=img_size,
        batch_size=batch_size,
    )
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_set.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    
filter_dataset()
generate_dataset((180, 180), 128)