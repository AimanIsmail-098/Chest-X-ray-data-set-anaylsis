import os
import shutil
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Load Model
# -----------------------------
model = load_model("pneumonia_cnn.keras")

# -----------------------------
# Load Test Dataset
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "Testing",
    target_size=(224,224),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(test_generator, verbose=0)

predicted = (predictions > 0.5).astype(int).flatten()
actual = test_generator.classes

class_names = list(test_generator.class_indices.keys())

file_paths = test_generator.filepaths

# -----------------------------
# Create Output Folder
# -----------------------------
output_folder = "Misclassified_Images"

os.makedirs(output_folder, exist_ok=True)

count = 0

for i in range(len(actual)):

    if predicted[i] != actual[i]:

        old_path = file_paths[i]

        filename = (
            f"{count+1}_Actual_{class_names[actual[i]]}"
            f"_Predicted_{class_names[predicted[i]]}.jpg"
        )

        new_path = os.path.join(output_folder, filename)

        shutil.copy(old_path, new_path)

        count += 1

        if count == 10:
            break

print(f"{count} misclassified images saved successfully.")