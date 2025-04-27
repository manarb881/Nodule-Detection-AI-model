!pip install ultralytics torch torchvision


import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Utility to parse XML for bounding box data
def parse_xml_to_dict(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        data = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            class_name = obj.find("name").text
            data.append({
                "image_name": root.find("filename").text,
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })
        return data
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return []

# Enhance image contrast using CLAHE
def enhance_contrast(image):
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    except Exception as e:
        print(f"Error enhancing contrast: {e}")
        return image

# Process a single image and save enhanced version + YOLO annotations
def process_image(image_path, annotations, output_dir):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or invalid format")

        # Enhance contrast
        enhanced_image = enhance_contrast(image)

        # Save enhanced image
        image_filename = os.path.basename(image_path)
        enhanced_image_path = os.path.join(output_dir, "images", image_filename)
        cv2.imwrite(enhanced_image_path, enhanced_image)

        # Prepare YOLO format annotations
        h, w = image.shape
        yolo_annotations = []
        for bbox in annotations:
            x_center = ((bbox["xmin"] + bbox["xmax"]) / 2) / w
            y_center = ((bbox["ymin"] + bbox["ymax"]) / 2) / h
            bbox_width = (bbox["xmax"] - bbox["xmin"]) / w
            bbox_height = (bbox["ymax"] - bbox["ymin"]) / h
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Save annotations
        annotation_filename = os.path.splitext(image_filename)[0] + ".txt"
        annotation_path = os.path.join(output_dir, "labels", annotation_filename)
        with open(annotation_path, "w") as f:
            f.write("\n".join(yolo_annotations))

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Main processing function with DataFrame creation
def process_dataset(images_dir, annotations_dir, output_dir):
    try:
        # Create output directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

        # List of all bounding boxes for DataFrame
        all_bboxes = []

        # Process each XML file
        for xml_file in tqdm(os.listdir(annotations_dir), desc="Processing dataset"):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(annotations_dir, xml_file)
            annotations = parse_xml_to_dict(xml_path)

            # Append bounding boxes to list
            all_bboxes.extend(annotations)

            image_filename = os.path.splitext(xml_file)[0] + ".jpg"
            image_path = os.path.join(images_dir, image_filename)

            if os.path.exists(image_path):
                process_image(image_path, annotations, output_dir)
            else:
                print(f"Image file {image_filename} not found for annotation {xml_file}")

        # Create a DataFrame
        df = pd.DataFrame(all_bboxes)

        # Add width and height columns
        df["w"] = df["xmax"] - df["xmin"]
        df["h"] = df["ymax"] - df["ymin"]

       # Add image dimensions to DataFrame
image_dimensions = {}
for img_name in tqdm(df["image_name"].unique(), desc="Reading image dimensions"):
    img_path = os.path.join(images_dir, img_name)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            image_dimensions[img_name] = (img.shape[1], img.shape[0])  # (width, height)
        else:
            print(f"Warning: Unable to read image {img_name}.")
    else:
        print(f"Warning: Image file {img_name} does not exist.")

# Map dimensions to DataFrame
df["image_width"] = df["image_name"].map(lambda x: image_dimensions.get(x, (None, None))[0])
df["image_height"] = df["image_name"].map(lambda x: image_dimensions.get(x, (None, None))[1])

# Replace NaN dimensions with -1 for clarity (optional)
df["image_width"].fillna(-1, inplace=True)
df["image_height"].fillna(-1, inplace=True)
        # Save the DataFrame as CSV
        df.to_csv("df.csv", index=False)

        print("Dataset processing complete. DataFrame saved to dataset_with_bboxes.csv.")
    except Exception as e:
        print(f"Error processing dataset: {e}")

# Example usage
if __name__ == "__main__":
    images_dir = "/kaggle/input/pulmonary-nodule/train/jpg"
    annotations_dir = "/kaggle/input/pulmonary-nodule/train/anno"
    output_dir = "/kaggle/working/data"

    process_dataset(images_dir, annotations_dir, output_dir)




def display_contrast_images(image_paths, contrast_function, num_images=5):
    """
    Display images before and after applying the contrast enhancement function.
    
    :param image_paths: List of image paths to display.
    :param contrast_function: Function to apply contrast enhancement.
    :param num_images: Number of images to display.
    """
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(image_paths[:num_images]):
        # Read the original image
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"Error loading image: {image_path}")
            continue

        # Apply contrast enhancement
        enhanced_img = contrast_function(image_path)

        # Display original image
        plt.subplot(num_images, 2, i * 2 + 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Display enhanced image
        plt.subplot(num_images, 2, i * 2 + 2)
        plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
        plt.title("Enhanced Image")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
# Example usage
image_files = [os.path.join(train_img_dir, img) for img in os.listdir(train_img_dir) if img.endswith('.jpg')]
display_contrast_images(image_files, enhance_contrast, num_images=5)


# center x, center y
df['center_x'] = ((df['xmax']+df['xmin'])/2)/df['img_width']
df['center_y'] = ((df['ymax']+df['ymin'])/2)/df['img_height']


# 80% train and 20% test
img_df = pd.DataFrame(images,columns=['file'])
img_train = tuple(img_df.sample(frac=0.8)['file']) # shuffle and pick 80% of images

train_df = df.query(f'file in {img_train}')
eval_df = df.query(f'file in {img_eval}')

import os
from shutil import move
