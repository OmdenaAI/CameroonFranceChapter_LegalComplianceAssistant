import os
import cv2
import numpy as np
from pathlib import Path


def get_files(dir_path, ext=".png"):
    files = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isdir(file_path):
            files.extend(get_files(file_path, ext))
        else:
            if os.path.splitext(file_path)[1].lower() == ext:
                files.append(file_path)

    return files
        

def repair_lines(img):
    gauss_win_size = 1
    gauss_sigma = 3

    img_blur = cv2.GaussianBlur(img, (gauss_win_size,gauss_win_size),gauss_sigma)
    
    return img_blur


def clear_background(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 50
    black_mask = gray < threshold
    output = np.ones_like(img) * 255
    output[black_mask] = [0, 0, 0]
    img = repair_lines(output)

    cv2.imwrite(img_path, img)