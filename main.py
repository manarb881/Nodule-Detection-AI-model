import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics import YOLO  # YOLO model interface
import matplotlib.pyplot as plt
from shutil import move
import numpy as np

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

# Process dataset and prepare for training
def process_dataset(images_dir, annotations_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    all_bboxes = []
    for xml_file in tqdm(os.listdir(annotations_dir), desc="Processing dataset"):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        annotations = parse_xml_to_dict(xml_path)
        all_bboxes.extend(annotations)
        image_filename = os.path.splitext(xml_file)[0] + ".jpg"
        image_path = os.path.join(images_dir, image_filename)

        if os.path.exists(image_path):
            process_image(image_path, annotations, output_dir)
        else:
            print(f"Image file {image_filename} not found for annotation {xml_file}")

    # Create a DataFrame
    df = pd.DataFrame(all_bboxes)
    df.to_csv("dataset_with_bboxes.csv", index=False)

# Train YOLO model
def train_model(data_dir, model_path="yolov5s.pt", epochs=10):
    model = YOLO(model_path)
    model.train(data=os.path.join(data_dir, "data.yaml"), epochs=epochs)
    return model

# Inference and prediction
def predict_and_display(model, image_path):
    results = model.predict(source=image_path)
    img = cv2.imread(image_path)
    for result in results:
        bbox, conf, cls = result["bbox"], result["confidence"], result["class"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class: {cls}, Conf: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
