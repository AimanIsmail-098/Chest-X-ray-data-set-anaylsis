import os
import matplotlib.pyplot as plt
dataset_path = "dataset"
classes = []
counts = []

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        images = os.listdir(class_path)

        classes.append(class_name)
        counts.append(len(images))

plt.bar(classes, counts)

plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")

plt.show()