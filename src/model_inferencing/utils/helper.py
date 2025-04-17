from pdf2image import convert_from_path
from PIL import Image
import settings
from PIL import Image
import os
import shutil

def save_pdf(doc, output_path):
    output_path = output_path + "/Redacted_pdf.pdf"
    doc.save(output_path, garbage=4, deflate=True, incremental=False)
    doc.close()
    return "saved successfully"


def convert_pdf_to_images():
    os.makedirs(settings.IMAGE_PATH, exist_ok=True)
    pdf_list = [settings.OUTPUT_PDF_PATH+"/Redacted_pdf.pdf"]
    for pdfs in pdf_list:
        file_name_image = pdfs.split(r"/")[-1].split(".")[0]
        pages = convert_from_path(pdfs, 500)

        for count, page in enumerate(pages):

            page.save(rf'{settings.IMAGE_PATH}/{file_name_image}_pg{count}.jpg', 'JPEG')

def convert_img_to_pdf():
    image_files = os.listdir(f"{settings.YOLO_OUTPUT_FOLDER}/masked")
    image_paths = [f"{settings.YOLO_OUTPUT_FOLDER}/masked/{image_file}" for image_file in image_files]  # Adjust the range based on the number of images

    print(image_paths)
    first_image = Image.open(image_paths[0])

    other_images = [Image.open(img_path).convert("RGB") for img_path in image_paths[1:]]

    output_pdf_path = f"{settings.YOLO_OUTPUT_FOLDER}/Redacted.pdf"
    first_image.convert("RGB").save(output_pdf_path, save_all=True, append_images=other_images)
    shutil.rmtree(f"{settings.YOLO_OUTPUT_FOLDER}/masked", ignore_errors=True)
    shutil.rmtree(f"{settings.YOLO_SUBFOLDER}", ignore_errors=True)
    shutil.rmtree(settings.IMAGE_PATH)
    print(f"Combined PDF saved at: {output_pdf_path}")