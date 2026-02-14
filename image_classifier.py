import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

def filter_dataset():
    categories = ("Black Sea Sprat", "Hourse Mackerel", "Red Mullet", 
                  "Red Sea Bream", "Sea Bass", "Shrimp", 
                  "Striped Red Mullet", "Trout")
    for folder_name in categories:
        folder_path = os.path.join("Fish_Dataset", folder_name)
        
        if not os.path.exists(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.isdir(file_path):
                continue
            
            try:
                with open(file_path, "rb") as f:
                    is_png = b"PNG" in f.peek(10).upper()
                if not is_png:
                    print(f"Removing non-png file: {file_path}")
                    os.remove(file_path) # --- FIX 3: Remove path, not object ---
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
def generate_dataset(img_size, batch_size):
    class_names = ["Black Sea Sprat", "Hourse Mackerel", "Red Mullet", 
                   "Red Sea Bream", "Sea Bass", "Shrimp", 
                   "Striped Red Mullet", "Trout"]
    train_set, validation_set = keras.utils.image_dataset_from_directory(
        "Fish_Dataset",
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        validation_split=0.2, #20% is used for validation
        subset="both",
        seed = 1337,
        image_size=img_size,
        batch_size=batch_size,
    )
    
    #Plot 10 different images from the train dataset
    plt.figure(figsize=(10, 10))
    for images, labels in train_set.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
            
    #Plot image using data_augmentation
    plt.figure(figsize=(10, 10))
    for images, _ in train_set.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")

    #plt.show()
    return train_set, validation_set


#Add random rotations and horizontal flips to help expose the model to different aspects of the training data
data_augmentation_layers = [layers.RandomFlip("horizontal"),
                                layers.RandomRotation(0.1)]
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    
    return images


def classification_model(input_shape, num_classes):
    #define the shape of the raw data
    inputs = keras.Input(shape = input_shape)
    
    #Create the entry block
    x = layers.Rescaling(1.0/255)(inputs) #normalize the image 0-1 instead of 0-255
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x) #Looks for 128 different features in the image
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x) #we use the relu activation function to learn non linear patterns
    
    previous_block_activation = x
    
    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

if __name__ == "__main__":
    filter_dataset()
    img_size = (180, 180)
    train_set, val_set = generate_dataset(img_size, 128)

    #standardized_dataset = train_set.map(lambda x, y: ( data_augmentation(x), y))

    #Apply augmentation to all images in training dataset
    train_Set = train_set.map(
        lambda img, label:(data_augmentation(img), label),
        num_parallel_calls = tf_data.AUTOTUNE
    )
    
    model = classification_model(input_shape=img_size + (3,), num_classes=8)
    keras.utils.plot_model(model, show_shapes=True)
    
    #train the model
    epochs = 25

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    
    model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    
    model.fit(
        train_set,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_set,
    )

