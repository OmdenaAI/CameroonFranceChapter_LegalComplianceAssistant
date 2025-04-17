# Path constants
INPUT_PDF_PATH = r"/app/output/input.pdf"
OUTPUT_PDF_PATH = r"/app/output"
IMAGE_PATH = r"/app/output/images"

# Gliner constansts
LABELS = ["Company Name", "Person", "Money", "Date", "Tenure", 
          "Country", "Street Address", "STREET ADDRESS", "State", "City",
          "Medicine Name","ID","Phone"]
PRIORITY_LABELS = ["Street Address", "STREET ADDRESS", "State", "City", "Country"]
GLINER_MODEL = "/app/gliner_inference_model" # self finetuned model
THRESHOLD = 0.75
FONT_NAME = "Helvetica"

LABELS_MAPPING = {"Company Name" : "Com",
                  "Person": "P",
                  "Money": "$",
                  "Date": "Date",
                  "Tenure":"Ten",
                  "Country":"Co",
                  "Street Address":"Addr",
                  "STREET ADDRESS":"Addr",
                  "State":"St",
                  "City":"Ct",
                  "Medicine Name":"Med",
                  "ID":"ID",
                  "Phone":"Ph"}
# YOLO Path
PYTHON_EXEC = "python"
YOLO_DETECT_PATH = "yolov5/detect.py"
YOLO_WEIGHTS = "yolov5/runs/train/exp/weights/best.pt"
YOLO_OUTPUT_FOLDER = "/app/output"
YOLO_SUBFOLDER = "/app/output/exp"
