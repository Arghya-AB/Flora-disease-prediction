import tensorflow as tf
import random
import numpy as np
import os
import pathlib

if not os.path.exists("datasets"):
    os.makedirs("datasets")

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00486/rice_leaf_diseases.zip"
data_dir = tf.keras.utils.get_file(
    "rice_leaf_diseases",
    origin=dataset_url,
    extract=True,
    cache_dir=".",
)
data_dir = pathlib.Path("datasets")
class_names = ["bacterial leaf blight", "brown spot", "leaf smut"]
batch_size = 12
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=batch_size,
)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=batch_size,
)

# Normalizing the pixel values - apply to both train and validation set
normalization_layer = tf.keras.Sequential(
    [tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)]
)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"
        ),  # Flip along both axes
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            0.1
        ),  # Randomly zoom images in dataset
    ]
)

print("Train size (number of batches) before augmentation: ", len(train_ds))
# Apply only to train set
aug_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
print("Size (number of batches) of augmented dataset: ", len(aug_ds))
# Adding to train_ds
train_ds = train_ds.concatenate(aug_ds)
print("Train size (number of batches) after augmentation: ", len(train_ds))
base_model = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(256, 256, 3), include_top=False
)  # False, do not include the classification layer of the model


base_model.trainable = False
inputs = tf.keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(
    x
)  # Add own classififcation layer
model = tf.keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
epochs = 15
history = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs, use_multiprocessing=True
)
model.summary()
model.save("models")
x = random.randint(0, batch_size - 1)
for i in val_ds.as_numpy_iterator():
    img, label = i
    output = model.predict(
        np.expand_dims(img[x], 0)
    )  # getting output; input shape (256, 256, 3) --> (1, 256, 256, 3)
    pred = np.argmax(output[0])  # finding max
    print(
        "Prdicted: ", class_names[pred]
    )  # Picking the label from class_names base don the model output
    print("True: ", class_names[label[x]])
    print("Probability: ", output[0][pred])
    break
