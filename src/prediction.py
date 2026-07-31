import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("pneumonia_cnn.keras")

# -----------------------------
# Load Test Dataset
# -----------------------------
test_path = "Testing"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# -----------------------------
# Make Predictions
# -----------------------------
predictions = model.predict(test_generator)

# Convert probabilities to 0 and 1
predicted_classes = (predictions > 0.5).astype(int).flatten()

# True Labels
true_classes = test_generator.classes

# -----------------------------
# Performance Metrics
# -----------------------------
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)

print("\n========== PERFORMANCE METRICS ==========\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix\n")
print(cm)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=test_generator.class_indices.keys()
    )
)

# -----------------------------
# Plot Confusion Matrix
# -----------------------------
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys()
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()