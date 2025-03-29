import torch
import joblib
import consts
import torchvision
import pytesseract
from PIL import Image
from torchvision.transforms import v2
from torch.nn import Sequential, Linear
from nlp_utils import get_text_similarity
from images_preprocessing import clear_background
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-large-handwritten"


def analyze_print(img_paths):
    processor = get_processor(MODEL_NAME)
    model = get_vision_model(MODEL_NAME, processor)
    images_text = {}

    for img_path in img_paths:
        image, ocr_data = run_ocr(img_path)
        text_bboxes = get_text_bboxes(image, ocr_data, model, processor)
        text_lines = get_text_lines(text_bboxes)
        images_text[img_path] = {
            "image_size": image.size,
            "text_lines": text_lines
        }

    return images_text


def extract_features(img):
    vgg16 = torchvision.models.vgg16()
    feature_extractor = Sequential(*list(vgg16.children())[:-1])
    projection_layer = Linear(512 * 7 * 7, 512)

    with torch.no_grad():
        features = feature_extractor(img)
        features = torch.flatten(features, start_dim=1)
        features = projection_layer(features)
        
        return features


def get_image_transformations(mean, std):
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean, std)
    ])

    return transform


def analyze_handwritting(img_paths):
    transform = get_image_transformations(consts.HANDWRITTEN_SIGNATURES_MEAN, 
                                          consts.HANDWRITTEN_SIGNATURES_STD)
    results = []
    model = joblib.load(consts.HANDWRITTEN_SIGNATURES_MODEL_PATH)
    
    for img_path in img_paths:
        clear_background(img_path)
        img = Image.open(img_path)
        img_tensor = transform(img)
        batch = img_tensor.unsqueeze(0)
        features = extract_features(batch).cpu().numpy()
        y_pred = model.predict(features)
        results.append((img_path, bool(y_pred)))

    return results


def get_processor(model_name):
    processor = TrOCRProcessor.from_pretrained(model_name)
    return processor


def get_vision_model(model_name, processor, num_return_sequences=3):
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 50
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.config.num_return_sequences = num_return_sequences 

    return model


def analyze_print_from_image(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generate_result = model.generate(pixel_values, 
                                     output_scores=True, 
                                     #num_return_sequences=num_return_sequences, 
                                     return_dict_in_generate=True)
    ids, scores = generate_result['sequences'], generate_result['sequences_scores']
    generated_text = processor.batch_decode(ids, skip_special_tokens=True)

    return generated_text, scores


def are_textboxes_adjacent(text_box, text_boxes):
    for tb in text_boxes:
        if (abs(tb[2] - text_box[0]) < 10 and abs(tb[1] - text_box[1]) < 5) or \
           (abs(tb[0] - text_box[0]) < 5 and abs(tb[3] - text_box[1]) < 10):
            return True
    
    return False


def run_ocr(img_path):
    image = Image.open(img_path).convert("RGB")
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    return image, ocr_data


def get_text_bboxes(image, 
                    ocr_data, 
                    vision_model, 
                    image_processor,
                    include_empty_text=True,
                    score_threshold=-0.05, 
                    text_similarity_threshold=0.95):
    text_bboxes = []

    for idx, ocr_text in enumerate(ocr_data["text"]):
        if ocr_text.strip() != "" or include_empty_text:
            text_area = [
                ocr_data["left"][idx], 
                ocr_data["top"][idx], 
                ocr_data["left"][idx] + ocr_data["width"][idx], 
                ocr_data["top"][idx] + ocr_data["height"][idx]
            ]
            cropped_img = image.crop(text_area)
            generated_texts, scores = analyze_print_from_image(cropped_img, vision_model, image_processor)
            max_score_index = torch.argmax(scores).item()
            if scores[max_score_index].item() > score_threshold:
                best_text = None
                best_similarity_score = 0.0
                for gen_text in generated_texts:
                    text_similarity_score = get_text_similarity(gen_text, ocr_text)
                    if text_similarity_score > text_similarity_threshold and \
                        text_similarity_score > best_similarity_score:
                        best_similarity_score = text_similarity_score
                        best_text = gen_text
                if best_text:
                    text_bboxes.append({ "text": best_text, "bbox": text_area })

    return text_bboxes


def get_enclosing_bbox(bboxes):
    bbox = [
        min(b["bbox"][0] for b in bboxes), 
        min(b["bbox"][1] for b in bboxes),
        max(b["bbox"][2] for b in bboxes),
        max(b["bbox"][3] for b in bboxes)
    ]

    return bbox


def get_text_lines(text_bboxes):
    if not text_bboxes:
        return None

    lines = []

    for i in range(len(text_bboxes) - 1):
        related_text_boxes = [text_bboxes[i]]
        for j in range(i + 1, len(text_bboxes)):
            if are_textboxes_adjacent(text_bboxes[j]["bbox"], [t["bbox"] for t in related_text_boxes]):
                related_text_boxes.append(text_bboxes[j])
        bbox = get_enclosing_bbox(related_text_boxes)
        lines.append({ 
            "text": " ".join([t["text"] for t in related_text_boxes]),
            "bbox": bbox
        })

    return get_unique_lines(lines)


def get_unique_lines(lines):
    sorted_lines = list(sorted(lines, key=lambda i: len(i["text"]), reverse=True))
    unique_lines = []

    for l1 in sorted_lines:
        is_unique = True
        for l2 in unique_lines:
            if l1["text"] in l2["text"]:
                is_unique = False
                break
        if is_unique:
            unique_lines.append(l1)
    
    return unique_lines