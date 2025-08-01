import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pytesseract
from typing import Tuple
import numpy as np
from PIL import Image, ExifTags, ImageOps
from imutils.object_detection import non_max_suppression
from paddleocr import PaddleOCR
from torchvision import transforms
import re
import string
from hashlib import sha1

home: Path = Path.home()
ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=False, # Disables text line orientation classification
    doc_orientation_classify_model_dir=f"{home}/.paddlex/official_models/PP-LCNet_x1_0_doc_ori",  # remove these if running for the first time
    doc_unwarping_model_dir=f"{home}/.paddlex/official_models/UVDoc",  # remove these if running for the first time
    text_detection_model_dir=f"{home}/.paddlex/official_models/PP-OCRv5_server_det",  # remove these if running for the first time
    text_recognition_model_dir=f"{home}/.paddlex/official_models/PP-OCRv5_server_rec",  # remove these if running for the first time
    textline_orientation_model_dir=f"{home}/.paddlex/official_models/PP-LCNet_x1_0_textline_ori",  # remove these if running for the first time
    text_det_unclip_ratio=2
)

MISTER_COLOR: str = "Mr. Color"

GLOSS_JP: str = "光沢"
GLOSS_EN: str = "Gloss"

SEMI_GLOSS_JP: str = "半光沢"
SEMI_GLOSS_EN: str = "Semi Gloss"

FLAT_JP: str = "つや消し"
FLAT_EN: str = "Flat"

UNKNOWN: str = "Unknown"

def get_top_left(box):
    # box of format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    return box[0][0], box[0][1]


def sort_ocr_results(ocr_results, y_threshold=10) -> list[dict]:
    """Sort OCR results from top left to bottom right."""
    elements = []
    for idx, (text, box, score) in enumerate(zip(ocr_results["rec_texts"], ocr_results["rec_polys"], ocr_results["rec_scores"])):
        x, y = get_top_left(box)
        elements.append({"x": x,  "y": y, "text": text, "score": score, "box": box, "index": idx})

    elements.sort(key=lambda e: (round(e["y"] / y_threshold), e["x"]))
    return elements


def flexible_pattern(match: str) -> str:
    """Make regex pattern that is based on a given string, ignoring space, punctuation."""
    base: str = re.sub(r"[\s{}]+".format(re.escape(string.punctuation)), "", match.lower())
    pattern: str = ""
    for char in base:
        pattern += re.escape(char) + r"[\sP{}]*".format(re.escape(string.punctuation))
    return pattern


def flexible_match(pattern: str, text: str) -> str:
    """Flexibly match text, ignore punctuation, case."""
    regex: str = flexible_pattern(pattern)
    return re.search(regex, text, re.IGNORECASE)


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

    parsed_colors = {}

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
        result = ocr.predict(opening, text_rec_score_thresh=0.8, use_doc_unwarping=True)
        sorted_results = sort_ocr_results(result[0])  # Only ever using a one element list
        manufacturer: str = UNKNOWN
        color: str = UNKNOWN
        finish: str = UNKNOWN
        for result in sorted_results:
            text: str = result["text"].lower()
            if flexible_match(MISTER_COLOR, text):
                manufacturer = MISTER_COLOR
            
            # When mister color is verified, look for finish and color
            if manufacturer is MISTER_COLOR: 
                if flexible_match(GLOSS_JP, text):
                    finish=GLOSS_EN
                elif flexible_match(SEMI_GLOSS_JP, text):
                    finish=SEMI_GLOSS_EN
                elif flexible_match(FLAT_JP, text):
                    finish=FLAT_EN
                
                # When we know the finish, we're near the end of the label
                # What remains should be the color
                elif finish is not UNKNOWN:
                    if color is UNKNOWN:
                        color = text
                    elif "gloss" in text or "flat" in text:
                        # english paint finishes show up in this area too, skip them 
                        continue
                    else:
                        color = f"{color} {text}"
        
        hash_string: str = manufacturer+color+finish
        paint_hash: int = int(sha1(hash_string.encode("utf-8")).hexdigest(), 16) % 10 ** 8
        parsed_colors[paint_hash] = {
            "manufacturer": manufacturer,
            "color": color,
            "finish": finish
        }

    print(parsed_colors)

                

if __name__ == "__main__":
    main()