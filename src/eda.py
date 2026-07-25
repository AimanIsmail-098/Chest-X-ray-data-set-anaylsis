#TASK 1

import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Total Images
images = []

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):

        for file_name in os.listdir(class_path):

            if file_name.lower().endswith(IMAGE_EXTENSIONS):

                image_path = os.path.join(class_path, file_name)

                images.append({
                    "path": image_path,
                    "class": class_name
                })


total_images = len(images)

print("DATASET ANALYSIS")

print("\nTotal Number of Images:", total_images)


#Images per class
class_counts = Counter()

for image in images:

    class_counts[image["class"]] += 1


print("\nImages Per Class:")

for class_name, count in class_counts.items():

    print(class_name, ":", count)

# Class Imabalance
largest_class = max(class_counts.values())
smallest_class = min(class_counts.values())

imbalance_ratio = largest_class / smallest_class

print("\nClass Imbalance Analysis")

print("Largest Class:", largest_class)
print("Smallest Class:", smallest_class)

print(
    "Imbalance Ratio:",
    round(imbalance_ratio, 2),
    ": 1"
)

# Image Dimensions
dimensions = []

corrupted_images = []


for image in images:

    try:

        with Image.open(image["path"]) as img:

            width, height = img.size

            dimensions.append((width, height))


    except Exception:

        corrupted_images.append(image["path"])


print("\nImage Dimension Analysis")

print("Valid Images:", len(dimensions))

print("Corrupted Images:", len(corrupted_images))



dimension_counts = Counter(dimensions)


print("\nMost Common Image Dimensions:")

for dimension, count in dimension_counts.most_common(10):

    print(
        dimension[0],
        "x",
        dimension[1],
        ":",
        count,
        "images"
    )


sample_images = []

# Sample Visualization
for class_name in class_counts:

    class_images = [

        image for image in images

        if image["class"] == class_name

    ]

    sample_images.extend(class_images[:2])


plt.figure(figsize=(12, 8))


for i, image in enumerate(sample_images):

    try:

        img = Image.open(image["path"])

        plt.subplot(2, 4, i + 1)

        plt.imshow(img, cmap="gray")

        plt.title(image["class"])

        plt.axis("off")


    except Exception:

        print(
            "Could not display:",
            image["path"]
        )


plt.tight_layout()

plt.show()


## TASK 2
# import os
# from PIL import Image
# from collections import Counter
# import matplotlib.pyplot as plt

# DATASET_PATH = "dataset"

# IMAGE_EXTENSIONS = (
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".bmp",
#     ".tif",
#     ".tiff"
# )


# # 1. COLLECT IMAGE DATA

# images = []

# for class_name in os.listdir(DATASET_PATH):

#     class_path = os.path.join(
#         DATASET_PATH,
#         class_name
#     )

#     if os.path.isdir(class_path):

#         for file_name in os.listdir(class_path):

#             if file_name.lower().endswith(
#                 IMAGE_EXTENSIONS
#             ):

#                 image_path = os.path.join(
#                     class_path,
#                     file_name
#                 )

#                 images.append({
#                     "path": image_path,
#                     "class": class_name
#                 })


# # 2. CLASS DISTRIBUTION GRAPH

# class_counts = Counter()

# for image in images:

#     class_counts[image["class"]] += 1


# plt.figure(figsize=(10, 6))

# plt.bar(
#     class_counts.keys(),
#     class_counts.values()
# )

# plt.title(
#     "Class Distribution of Chest X-ray Dataset"
# )

# plt.xlabel("Classes")

# plt.ylabel("Number of Images")

# plt.xticks(rotation=45)

# plt.tight_layout()

# plt.show()

# # 3. SAMPLE IMAGE VISUALIZATION

# sample_images = []

# for class_name in class_counts:

#     class_images = [

#         image for image in images

#         if image["class"] == class_name

#     ]

#     sample_images.extend(
#         class_images[:2]
#     )

# plt.figure(figsize=(12, 8))

# for i, image in enumerate(sample_images):

#     try:

#         img = Image.open(
#             image["path"]
#         )

#         plt.subplot(
#             2,
#             4,
#             i + 1
#         )

#         plt.imshow(
#             img,
#             cmap="gray"
#         )

#         plt.title(
#             image["class"]
#         )

#         plt.axis("off")


#     except Exception:

#         print(
#             "Could not display:",
#             image["path"]
#         )


# plt.tight_layout()

# plt.show()


# # 4. IMAGE DIMENSION SUMMARY

# dimensions = []


# for image in images:

#     try:

#         with Image.open(
#             image["path"]
#         ) as img:

#             width, height = img.size

#             dimensions.append(
#                 (width, height)
#             )


#     except Exception:

#         pass


# # Count image dimensions

# dimension_counts = Counter(
#     dimensions
# )


# # Get the 10 most common dimensions

# most_common_dimensions = (
#     dimension_counts
#     .most_common(10)
# )


# dimension_labels = []

# dimension_values = []


# for dimension, count in most_common_dimensions:

#     width, height = dimension

#     dimension_labels.append(
#         f"{width}x{height}"
#     )

#     dimension_values.append(
#         count
#     )



# plt.figure(figsize=(12, 6))

# plt.bar(
#     dimension_labels,
#     dimension_values
# )

# plt.title(
#     "Most Common Image Dimensions"
# )

# plt.xlabel(
#     "Image Dimensions"
# )

# plt.ylabel(
#     "Number of Images"
# )

# plt.xticks(
#     rotation=45
# )

# plt.tight_layout()

# plt.show()
