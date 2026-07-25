import os

dataset_path = "dataset"

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        print("Class:", class_name)

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            print(image_path)