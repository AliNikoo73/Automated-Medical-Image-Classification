import cv2
import numpy as np

def segment_lungs(image):
  """
  Segments the lungs from a chest X-ray image.

  Args:
      image: A numpy array representing the chest X-ray image.

  Returns:
      A tuple containing:
          - segmented_lungs: A numpy array representing the segmented lungs (1 for lung, 0 for background).
          - original_with_mask: A numpy array representing the original image with the segmented lungs overlaid in green.
  """
  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply thresholding to isolate the lungs
  thresh = cv2.threshold(gray, LT, UT, cv2.THRESH_BINARY)[1]

  # Apply morphological operations to improve segmentation
  kernel = np.ones((5,5), np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

  # Find connected components and keep the largest one (assuming it's the lungs)
  contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_area = 0
  largest_contour = None
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > largest_area:
      largest_area = area
      largest_contour = cnt

  # Create a mask with the segmented lungs (1 for lung, 0 for background)
  segmented_lungs = np.zeros_like(closing)
  if largest_contour is not None:
    cv2.drawContours(segmented_lungs, [largest_contour], 0, 255, -1)

  # Overlay the segmented lungs mask on the original image (green color)
  original_with_mask = image.copy()
  contours_list = [largest_contour]  # Assuming only one lung contour is needed
  cv2.drawContours(original_with_mask, contours_list, 0, (0, 255, 0), 2)

  return segmented_lungs, original_with_mask

LT = 130
UT = 255
  
# Example usage
image = cv2.imread("/Users/baharmac/Documents/Github/Automated-Medical-Image-Classification/data/train/Normal/07.jpeg")
segmented_lungs, original_with_mask = segment_lungs(image)

cv2.imwrite(f"/Users/baharmac/Documents/Github/Automated-Medical-Image-Classification/data/masked-segmented/v1.0/segmented_lungs_Threshold {LT}_{UT}.png", segmented_lungs)
cv2.imwrite(f"//Users/baharmac/Documents/Github/Automated-Medical-Image-Classification/data/masked-segmented/v1.0/original_with_mask_Threshold {LT}_{UT}.png", original_with_mask)

# cv2.imshow("Original Image", image)
cv2.imshow("Segmented Lungs", segmented_lungs)
cv2.imshow("Original with Mask", original_with_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()