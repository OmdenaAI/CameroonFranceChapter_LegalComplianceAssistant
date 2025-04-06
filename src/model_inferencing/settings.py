# Path constants
INPUT_PDF_PATH = r"/app/output/input.pdf"
OUTPUT_PDF_PATH = r"/app/output"
IMAGE_PATH = r"/app/output/images"

# Gliner constansts
LABELS = ["Company Name", "Person", "Money", "Date", "Tenure", 
          "Country", "Street Address", "STREET ADDRESS", "State", "City",
          "Medicine Name","ID","Phone"]
PRIORITY_LABELS = ["Street Address", "STREET ADDRESS", "State", "City", "Country"]
#GLINER_MODEL = "urchade/gliner_multi_pii-v1"
#GLINER_MODEL = "gliner-community/gliner_large-v2.5"
GLINER_MODEL = "/app/gliner_inference_model" # self finetuned model
THRESHOLD = 0.75
FONT_NAME = "Helvetica"

# YOLO Path
PYTHON_EXEC = "python"
YOLO_DETECT_PATH = "yolov5/detect.py"
YOLO_WEIGHTS = "yolov5/runs/train/exp/weights/best.pt"
YOLO_OUTPUT_FOLDER = "/app/output"
YOLO_SUBFOLDER = "/app/output/exp"
