import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
import numpy as np
    

def process_image(image_path: Path, threshold1: int, threshold2: int):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    combined = np.hstack((gray, edges))
    return combined, gray


def update(val):
    th1 = th1_scale.get()
    th2 = th2_scale.get()
    combined, _ = process_image(image_paths[current_image[0]], th1, th2)
    if combined is not None:
        cv2.imshow('Gray | Canny', combined)


def next_image():
    current_image[0] = (current_image[0] + 1) % len(image_paths)
    update(None)


def prev_image():
    current_image[0] = (current_image[0] - 1) % len(image_paths)
    update(None)


if __name__ == "__main__":
    image_tests_path: Path = Path(__file__).parent.parent / "tests" / "paints"
    image_paths: list[Path | None] = [x for x in image_tests_path.glob('**/*') if x.is_file()]
    if not image_paths:
        print("No image files found in the specified directory.")
        exit()

    current_image = [0]  # Use a list to allow modification in nested functions

    root = tk.Tk()
    root.title("Canny Threshold Adjustment")

    th1_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold 1", command=update)
    th1_scale.set(100)
    th1_scale.pack(fill='x')
    th2_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold 2", command=update)
    th2_scale.set(200)
    th2_scale.pack(fill='x')
    
    btn_prev = tk.Button(root, text="Previous Image", command=prev_image)
    btn_prev.pack(side='left')
    btn_next = tk.Button(root, text="Next Image", command=next_image)
    btn_next.pack(side='right')

    update(None)
    root.mainloop()
    cv2.destroyAllWindows()