import subprocess
import settings

def image_inference_yolo():
    command = [
        settings.PYTHON_EXEC, settings.YOLO_DETECT_PATH,
        "--source", settings.IMAGE_PATH,  # Change to your image/video path
        "--weights", settings.YOLO_WEIGHTS,  # Change to your model path
        "--conf", "0.5",
        "--img-size", "4250", "5500",
        "--save-txt"
    ]

    subprocess.run(command)