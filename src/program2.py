import os

dataset_path = "dataset"

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        images = os.listdir(class_path)

        print(class_name, ":", len(images), "images")