import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification, AdamW
from datasets import load_dataset, Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

def iob_to_label(label):
    if not label:
        return 'other'
    return label

def unnormalize_box(bbox, width, height):
    """
    Convert normalized bounding box coordinates to absolute coordinates.
    """
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def expand_bounding_box(box, expand_pixels=5, width=None, height=None):
    """
    Expand a bounding box by a fixed number of pixels in all directions.
    Ensure the expanded box stays within the image boundaries.
    """
    return [
        max(0, box[0] - expand_pixels),  # x_min
        max(0, box[1] - expand_pixels),  # y_min
        min(width, box[2] + expand_pixels),  # x_max
        min(height, box[3] + expand_pixels)  # y_max
    ]

# Define label-to-color mapping
label2color = {'Signature': 'blue', 'Logo':'red', 'O': 'orange'}

# Loop through each query
final_true_boxes = []
i = 0
model = LayoutLMv2ForTokenClassification.from_pretrained(r"D:\Raghu Studies\omdena\CameroonFranceChapter_LegalComplianceAssistant\src\saved_model")
labels = ["O", "Signature","Logo"]
id2label = ["O", "Signature","Logo"]
id2label = {l:i for i, l in enumerate(id2label)}
label2color = {'Signature':'blue', "Logo":'red', 'O':'orange'}
for query_index in range(5):  # Adjust the range as needed
    query = dataset_dict['train'][query_index]
    image = Image.open(query['image_path'])
    image = image.convert("RGB")

    # Get image dimensions
    width, height = image.size

    # Process inputs for LayoutLM
    encoded_inputs = processor(
        image, query['words'], boxes=query['bboxes'], word_labels=query['ner_tags'],
        padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)

    outputs = model(**encoded_inputs)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    true_predictions = [id2label[prediction] for prediction in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Use a default font or load a custom one

    final_true_boxes.append(true_boxes)

    # Process each prediction and bounding box
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction)

        if predicted_label in ['Signature','Logo']:
            # Expand the bounding box slightly to ensure full coverage
            expanded_box = expand_bounding_box(box, expand_pixels=5, width=width, height=height)

            # Erase the original text by filling the bounding box with the background color
            draw.rectangle(expanded_box, fill="black")  # Replace "white" with the actual background color

    # Resize the image back to the original resolution (4250x5500)
    resized_image = image.resize((4250, 5500))

    # Save the modified and resized image
    resized_image.save(f"modified_image_{i}.png")
    i += 1