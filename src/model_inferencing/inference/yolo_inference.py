import subprocess
import settings

def image_inference_yolo():
    command = [
        settings.PYTHON_EXEC, settings.YOLO_DETECT_PATH,
        "--source", settings.IMAGE_PATH,  
        "--weights", settings.YOLO_WEIGHTS, 
        "--conf", "0.5",
        "--img-size", "4250", "5500",
        "--save-txt",
        "--project",settings.YOLO_OUTPUT_FOLDER,
        "--name", settings.YOLO_SUBFOLDER
    ]

    subprocess.run(command)