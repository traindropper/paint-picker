from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from torchvision import transforms
import re
import string
from models import PaintDTO, Base
from base_classes import FinishEnum, PaintMediumEnum, ManufacturerEnum
import logging
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, session as session_utils
from update_db import upsert_paint
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

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
        LOGGER.warning("No instances detected in image, using default crop: %s", image_path)
        image = np.array(crop_transform(Image.fromarray(image)))

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
        if not check_image(str(image_path)):
            LOGGER.warning("%s is not a usable image, skipping...", image_path)
            continue

        opening: np.ndarray = load_and_smart_crop(image_path)    
        
        result = ocr.predict(opening, text_rec_score_thresh=0.8, use_doc_unwarping=True)
        sorted_results = sort_ocr_results(result[0])  # Only ever using a one element list
        manufacturer: ManufacturerEnum | None = None
        color: str | None = None
        finish: FinishEnum | None = None
        paint_medium: PaintMediumEnum | None = None
        for result in sorted_results:
            text: str = result["text"].lower()
            if flexible_match(MISTER_COLOR, text):
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
    
        if not color:
            LOGGER.warning("Failed to extract color for image: %s", image_path)
            continue

        paint_dto: PaintDTO = PaintDTO(manufacturer=manufacturer, color=color, finish=finish, paint_medium=paint_medium)
        LOGGER.info("Appended DTO for ingestion: %s:", paint_dto)
        paint_list.append(paint_dto)

    if paint_list:
        engine: Engine = create_engine("sqlite:///paintdb.sqlite3")
        Session = sessionmaker(bind=engine)
        session = Session()
        Base.metadata.create_all(engine)
        for paint in paint_list:
            upsert_paint(session, paint)
        session_utils.close_all_sessions()
