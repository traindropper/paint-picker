import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def main():
    image_tests_path: Path = Path(__file__).parent.parent / "tests" / "paints"
    image_paths: list[Path | None] = [x for x in image_tests_path.glob('**/*') if x.is_file()]
    if not image_paths:
        print("No image files found in the specified directory.")
        return
    Path.mkdir(image_tests_path.parent / "grey_images", exist_ok=True)
    for image_path in image_paths:
        print(f"Processing image: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path.name}")
            continue
        
        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        
        # Save the processed image (optional)
        output_path = image_tests_path.parent / "grey_images" / f"gray_{image_path.name}"
        cv2.imwrite(str(output_path), gray_image)
        print(f"Saved processed image to: {output_path}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Edges')
        plt.imshow(edges, cmap='gray')
        plt.show()

if __name__ == "__main__":
    main()