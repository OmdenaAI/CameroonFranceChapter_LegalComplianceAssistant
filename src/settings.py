# Path constants
INPUT_PDF_PATH = r"data/SEC/Amazon.pdf"
OUTPUT_PDF_PATH = r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/output"
IMAGE_PATH = r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/output/images"

# Gliner constansts
LABELS = ["Company Name", "Person", "Money", "Date", "Tenure", "Country", "Street_Address", "State", "City"]
PRIORITY_LABELS = ["Street_Address", "State", "City", "Country"]
#GLINER_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_MODEL = "gliner-community/gliner_large-v2.5"
THRESHOLD = 0.5
FONT_NAME = "Helvetica"

# YOLO Path
PYTHON_EXEC = "D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/.src_omdena_legal/Scripts/python.exe"
YOLO_DETECT_PATH = r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/src/yolov5/detect.py"
YOLO_WEIGHTS = r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/src/yolov5/runs/train/exp/weights/best.pt"
YOLO_OUTPUT_FOLDER = r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/output"
YOLO_SUBFOLDER = r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/output/exp"
