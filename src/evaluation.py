import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("pneumonia_cnn.keras")

print("Model loaded successfully!")

# -----------------------------
# Load Test Dataset
# -----------------------------
test_path = "Testing"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# -----------------------------
# Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_generator)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy*100:.2f}%")