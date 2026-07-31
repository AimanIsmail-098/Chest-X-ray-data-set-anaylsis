import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------------
# Load the trained model
# ----------------------------------
model = load_model("pneumonia_cnn.keras")

# ----------------------------------
# Load the test dataset
# ----------------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "Testing",
    target_size=(224, 224),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# ----------------------------------
# Make predictions
# ----------------------------------
predictions = model.predict(test_generator)

predicted_classes = (predictions > 0.5).astype(int).flatten()

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

filenames = test_generator.filepaths

# ----------------------------------
# Find correct and incorrect predictions
# ----------------------------------
correct = []
incorrect = []

for i in range(len(true_classes)):
    if predicted_classes[i] == true_classes[i]:
        correct.append(i)
    else:
        incorrect.append(i)

print(f"\nCorrect Predictions : {len(correct)}")
print(f"Incorrect Predictions : {len(incorrect)}")

# ----------------------------------
# Show Correct Predictions
# ----------------------------------
plt.figure(figsize=(15,8))

for i, index in enumerate(random.sample(correct, min(5, len(correct)))):

    image = plt.imread(filenames[index])

    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.axis("off")

    plt.title(
        f"Correct\nActual: {class_labels[true_classes[index]]}\nPredicted: {class_labels[predicted_classes[index]]}",
        fontsize=9
    )

# ----------------------------------
# Show Incorrect Predictions
# ----------------------------------
for i, index in enumerate(random.sample(incorrect, min(5, len(incorrect)))):

    image = plt.imread(filenames[index])

    plt.subplot(2,5,i+6)
    plt.imshow(image)
    plt.axis("off")

    plt.title(
        f"Incorrect\nActual: {class_labels[true_classes[index]]}\nPredicted: {class_labels[predicted_classes[index]]}",
        fontsize=9
    )

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.show()

print("\nSample predictions saved as sample_predictions.png")