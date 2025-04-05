import fitz
import gliner
import settings
from inference.gliner_inference import replace_text_with_labels
from inference.yolo_inference import image_inference_yolo
from utils.helper import save_pdf
from utils.helper import convert_pdf_to_images
from utils.yolo_utils import create_black_bounding_box
from utils.helper import convert_img_to_pdf

def run():
    # # --- Initialize GLiNER ---
    model = gliner.GLiNER.from_pretrained(settings.GLINER_MODEL)
    # Replace text with labels in the input PDF
    doc = replace_text_with_labels(model, settings.INPUT_PDF_PATH)
    # Save the modified PDF
    save_pdf(doc, settings.OUTPUT_PDF_PATH)
    print(f"Modified PDF saved as {settings.OUTPUT_PDF_PATH}")
    # For YOLO
    convert_pdf_to_images()
    image_inference_yolo()
    create_black_bounding_box()
    convert_img_to_pdf()
    




