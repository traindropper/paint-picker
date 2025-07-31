import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pytesseract
from typing import Tuple
import numpy as np
from PIL import Image, ExifTags, ImageOps
from imutils.object_detection import non_max_suppression
from paddle import to_tensor
from paddleocr import PaddleOCR
import gc
from torchvision import transforms

ocr = PaddleOCR(
    use_doc_orientation_classify=True, # Disables document orientation classification model via this parameter
    use_doc_unwarping=True, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
)
ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter


def check_image(file_path: str) -> bool:
    """Ensure the file can be opened and read with PIL and OpenCV."""
    # Check PIL.
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception:
        return False
    
    # Check OpenCV
    img_cv2 = cv2.imread(file_path)
    if img_cv2 is None:
        return False
    
    return True


def main():
    image_tests_path: Path = Path(__file__).parent.parent / "tests" / "paints"
    image_paths: list[Path | None] = [x for x in image_tests_path.glob('**/*') if x.is_file()]
    if not image_paths:
        print("No image files found in the specified directory.")
        return
    # Perhaps i can train to hone in on this label
    crop_transform = transforms.CenterCrop((2000, 2000))  # height, width

    for image_path in image_paths:
        if not check_image(str(image_path)):
            print(f"{image_path.name} is not a usable image, skipping...")
            continue

        print(f"Processing image: {image_path.name}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_transform(Image.fromarray(image))
        gray_image: np.ndarray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        blurred_image: np.ndarray = cv2.GaussianBlur(gray_image, (3,3), 0)
        thresholded_image: np.ndarray = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        if len(thresholded_image.shape) == 2:
            thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        opening: np.ndarray = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=1)
        
        
        cv2.imwrite(f"{image_path.stem}_preprocessed.jpg", opening)
        # new_height: int = 4000
        # new_width: int = 4000
        # resized_image = cv2.resize(image, (new_height, new_width))
        result = ocr.predict(opening)
        for res in result:
            res.save_to_json(f"{image_path.stem}_output")  
        gc.collect()  # Save my precious memory!

if __name__ == "__main__":
    main()