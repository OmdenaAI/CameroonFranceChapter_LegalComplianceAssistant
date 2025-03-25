from pdf2image import convert_from_path
from pdf2image import convert_from_path
from PIL import Image
import settings

def save_pdf(doc, output_path):
    doc.save(output_path, garbage=4, deflate=True, incremental=False)
    doc.close()
    return "saved successfully"



def convert_pdf_to_images():
    pdf_list = [settings.OUTPUT_PDF_PATH]
    for pdfs in pdf_list:
        file_name_image = pdfs.split(r"/")[-1].split(".")[0]
        pages = convert_from_path(pdfs, 500)

        for count, page in enumerate(pages):

            page.save(rf'{settings.IMAGE_PATH}/{file_name_image}_pg{count}.jpg', 'JPEG')
