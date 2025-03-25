import cv2
import os
import settings

def create_black_bounding_box():
    source_dir = settings.YOLO_SUBFOLDER  # Directory containing detected images
    labels_dir = f"{settings.YOLO_SUBFOLDER}/labels"  # Directory containing detection .txt files
    output_dir = f"{settings.YOLO_OUTPUT_FOLDER}/masked"  # Directory to save modified images
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(source_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Skip non-image files
            continue
        
        image_path = os.path.join(source_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            # Draw black rectangles for each detection
            for line in lines:
                parts = line.strip().split()
                
        
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                x1 = int((x_center - bbox_width / 2) * w)
                y1 = int((y_center - bbox_height / 2) * h)
                x2 = int((x_center + bbox_width / 2) * w)
                y2 = int((y_center + bbox_height / 2) * h)
                
        
                label_height_extension = int(bbox_height * 0.5)  
                y1_extended = max(0, y1 - label_height_extension)  
                
                
                cv2.rectangle(img, (x1, y1_extended), (x2, y2), (0, 0, 0), thickness=-1)  # thickness=-1 fills the rectangle


        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, img)
    return "Black masks applied and modified images saved successfully!"