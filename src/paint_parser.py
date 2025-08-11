from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from paddleocr import PaddleOCR
from torchvision import transforms
import re
import string
from src.models import PaintDTO, Base
from src.base_classes import FinishEnum, PaintMediumEnum, ManufacturerEnum
import logging
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, session as session_utils
from src.update_db import upsert_paint
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo 
from src.helpers import get_font_path
import gc

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
    text_det_unclip_ratio=2,
)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25  # Set a custom testing threshold
predictor = DefaultPredictor(cfg)   

MISTER_COLOR: str = "Mr. Color"
AQUEOUS: str = "AQUEOUS"

GLOSS_JP: str = "光沢"
GLOSS_EN: str = "Gloss"

SEMI_GLOSS_JP: str = "半光沢"
SEMI_GLOSS_EN: str = "Semi Gloss"

FLAT_JP: str = "つや消し"
FLAT_EN: str = "Flat"

LOGGER: logging.Logger = logging.getLogger(__name__)


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


def simple_string(text: str) -> str:
    """Strip all non-simple english/latin/arabic numeral characters from a string, space aware."""
    pattern: str = r" *[^A-Za-z0-9\'\- ]+ *"
    cleaned: str = re.sub(pattern, " ", text)
    return cleaned.strip()


def plot_ocr_boxes(image: np.ndarray, ocr_results: list[dict], save_path: Path) -> None:
    """Plot OCR boxes and text on the image using PIL for consistency."""
    # Convert to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # support for Japanese characters
    font_size = 24
    font = ImageFont.truetype(get_font_path(), font_size)

    for result in ocr_results:
        box = result["box"]
        # box_points = [(int(x), int(y)) for x, y in box]
        # Draw polygon (box)
        # draw.line(box_points + [box_points[0]], fill=(0, 255, 0), width=2)
        # Draw text above the top-left corner
        top_left = get_top_left(box)
        text: str = f"{result['text']} ({round(result['score'], 3)})" 
        
        # Draw rectangle around the text
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_position = (top_left[0], top_left[1] - font_size - 4)
        x = text_position[0]
        y = text_position[1]

        # Add margin around the rectangle
        margin = 4
        rect_coords = [x - margin, y, x + text_w + margin, y + text_h + margin + 4]
        draw.rectangle(rect_coords, fill=(0, 0, 0, 180))  # semi-transparent black background
        
        draw.text(
            text_position,
            text,
            font=font,
            fill=(255, 255, 255, 255)  # white text with full opacity
        )

    # Save as BGR for OpenCV compatibility
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
    cv2.imwrite(str(save_path), result_img)
    LOGGER.info("Saved OCR boxes to %s", save_path.name)

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


def load_and_smart_crop(image_path: Path) -> np.ndarray:
    """Attempt to smart crop the image using Detectron2h """
    crop_transform = transforms.CenterCrop((2000, 2000))  # height, width
    image = cv2.imread(image_path)
    outputs = predictor(image)
    instances = outputs["instances"]
    if len(instances) == 0:
        LOGGER.warning("No instances detected in image, using uncropped image: %s", image_path)
        # image = np.array(crop_transform(Image.fromarray(image)))

    else:
        # Get the bounding box of the first instance
        box = instances[0].pred_boxes.tensor[0].cpu().numpy()
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        
        image = image[y1:y2, x1:x2]
        LOGGER.info("Smart cropped: %s", image_path.name)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    return opening


def parse_test_directory():
    image_tests_path: Path = Path(__file__).parent.parent / "tests" / "paints"
    image_paths: list[Path | None] = [x for x in image_tests_path.glob('**/*') if x.is_file()]
    if not image_paths:
        LOGGER.warning("No image files found in the specified directory: %s", image_tests_path)
        return  

    paint_list: list[PaintDTO] = []

    for image_path in image_paths:
        paint_dto: PaintDTO | None = parse_image(image_path)
        paint_list.append(paint_dto)

    if paint_list:
        engine: Engine = create_engine("sqlite:///paintdb.sqlite3")
        Session = sessionmaker(bind=engine)
        session = Session()
        Base.metadata.create_all(engine)
        for paint in paint_list:
            upsert_paint(session, paint)
        session_utils.close_all_sessions()


def parse_image(image_path: Path, save_ocr_path: Path | None = None) -> PaintDTO | None:
    """Parse a single image and return a PaintDTO."""
    if not check_image(str(image_path)):
        LOGGER.warning("%s is not a usable image, skipping...", image_path)
        return None
    
    opening: np.ndarray = load_and_smart_crop(image_path)    

    result = ocr.predict(opening, text_rec_score_thresh=0.8, use_doc_unwarping=True)
    sorted_results = sort_ocr_results(result[0])  # Only ever using a one element list
    if save_ocr_path:
        plot_ocr_boxes(opening, sorted_results, save_ocr_path)

    if not sorted_results:
        LOGGER.warning("No OCR results found in %s", image_path.name)
        return None

    # Initialize paint DTO
    manufacturer: ManufacturerEnum | None = None
    color: str | None = None
    finish: FinishEnum | None = None
    paint_medium: PaintMediumEnum | None = None

    for result in sorted_results:
        text: str = result["text"].lower()
        if flexible_match(MISTER_COLOR, text) or flexible_match(AQUEOUS, text) or flexible_match("hobby", text):
            manufacturer = ManufacturerEnum.MR_HOBBY
            paint_medium = PaintMediumEnum.LACQUER
        
        # When mister color is verified, look for finish and color
        if manufacturer is ManufacturerEnum.MR_HOBBY: 
            if flexible_match(SEMI_GLOSS_JP, text):  # Must check semi-gloss first. it contains substring gloss
                finish=FinishEnum.SEMI_GLOSS
            elif flexible_match(GLOSS_JP, text):
                finish=FinishEnum.GLOSS
            elif flexible_match(FLAT_JP, text):
                finish=FinishEnum.MATTE
            
            # When we know the finish, we're near the end of the label
            # What remains should be the color
            elif finish:
                if color:
                    if manufacturer is ManufacturerEnum.MR_HOBBY:
                        if "gloss" in text or "flat" in text or "semi" in text or "primary" in text:
                            # english paint finishes/primaries show up in this area too, skip them 
                            continue
                        else:
                            color = f"{color} {text}"
                    else:
                        color = f"{color} {text}"
                else:
                    color = text
    gc.collect()
    if color:
        color = simple_string(color)
    else:
        color = "UNKNOWN"
    return PaintDTO(manufacturer=manufacturer, color=color, swatch=None, finish=finish, paint_medium=paint_medium)

def parse_image_as_string(image_path: Path, save_ocr_path: Path | None = None) -> dict[str, str | None]:
    """Parse an image and return a dictionary with paint details as strings."""
    paint_dto: PaintDTO | None = parse_image(image_path, save_ocr_path=save_ocr_path)
    if not paint_dto:
        return {"manufacturer": None, "color": None, "swatch": None, "finish": None, "paint_medium": None}

    return {
        "manufacturer": paint_dto.manufacturer.value if paint_dto.manufacturer else None,
        "color": paint_dto.color,
        "swatch": paint_dto.swatch,
        "finish": paint_dto.finish.value if paint_dto.finish else None,
        "paint_medium": paint_dto.paint_medium.value if paint_dto.paint_medium else None
    }