import tensorflow as tf
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================
# Paths
# ==========================

TEST_DIR = "dataset"

CNN_MODEL = "models/pneumonia_cnn.keras"
MOBILE_MODEL = "models/mobilenet_model.keras"
RESNET_MODEL = "models/resnet_model.keras"

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# ==========================
# Test Dataset
# ==========================

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

# ==========================
# Function
# ==========================

def evaluate_model(model_path):

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_data, verbose=0)

    y_pred = (predictions > 0.5).astype(int).flatten()

    y_true = test_data.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# ==========================
# Evaluate Models
# ==========================

cnn = evaluate_model(CNN_MODEL)

mobile = evaluate_model(MOBILE_MODEL)

resnet = evaluate_model(RESNET_MODEL)

# ==========================
# Comparison Table
# ==========================

comparison = pd.DataFrame({

    "Model":[
        "CNN",
        "MobileNetV2",
        "ResNet50"
    ],

    "Accuracy":[
        cnn[0],
        mobile[0],
        resnet[0]
    ],

    "Precision":[
        cnn[1],
        mobile[1],
        resnet[1]
    ],

    "Recall":[
        cnn[2],
        mobile[2],
        resnet[2]
    ],

    "F1-score":[
        cnn[3],
        mobile[3],
        resnet[3]
    ]

})

print("\nComparison Table\n")

print(comparison)

comparison.to_csv("comparison_results.csv", index=False)

print("\ncomparison_results.csv saved successfully.")