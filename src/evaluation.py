import tensorflow as tf
import numpy as np

# ==========================
# Paths
# ==========================

TEST_DIR = "dataset"

CNN_MODEL = "models/pneumonia_cnn.keras"
MOBILENET_MODEL = "models/mobilenet_model.keras"
RESNET_MODEL = "models/resnet_model.keras"

# ==========================
# Dataset
# ==========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Classes:", test_data.class_indices)

# ==========================
# Evaluation Function
# ==========================

def evaluate_model(model_path, model_name):

    print("\n===================================")
    print(f"Evaluating: {model_name}")
    print("===================================")

    model = tf.keras.models.load_model(model_path)

    results = model.evaluate(test_data, verbose=1)

    print()

    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

# ==========================
# CNN
# ==========================

evaluate_model(CNN_MODEL, "CNN")

# ==========================
# MobileNetV2
# ==========================

evaluate_model(MOBILENET_MODEL, "MobileNetV2")

# ==========================
# ResNet50
# ==========================

evaluate_model(RESNET_MODEL, "ResNet50")

print("\n===================================")
print("All Models Evaluated Successfully")
print("===================================")