import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pytesseract


def main():
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    print(path_to_tesseract)
    # Providing the tesseract executable
    # location to pytesseract library
    pytesseract.tesseract_cmd = path_to_tesseract

    image_tests_path: Path = Path(__file__).parent.parent / "tests" / "paints"
    image_paths: list[Path | None] = [x for x in image_tests_path.glob('**/*') if x.is_file()]
    if not image_paths:
        print("No image files found in the specified directory.")
        return

    for image_path in image_paths:
        print(f"Processing image: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path.name}")
            continue
        
        # Example processing: Convert to grayscale, blur, then threshold
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (3,3), 0)
        thresholded_image = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        # Clean up the image with opening (erode then dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        cv2.imshow("grayscale", invert)
        cv2.waitKey(0)
        text = pytesseract.image_to_string(invert, lang="eng")
        # Displaying the extracted text
        print(text)
        # edges = cv2.Canny(gray_image, 100, 200)

if __name__ == "__main__":
    main()