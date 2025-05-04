import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(200)
	cv2.destroyAllWindows()

def detect_traffic_light(img, min_area=100):
    """
    Detect traffic light color: red, green, or yellow.
    Returns:
        detected_color: 'red', 'green', 'yellow', or None
        bbox: ((x1, y1), (x2, y2)) bounding box of detected light, or None
    """

    # 1. Preprocessing
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Color Ranges
    # Red (split into two ranges)
    lower_red1 = np.array([0, 70, 180])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 180])
    upper_red2 = np.array([180, 255, 255])

    # Green
    lower_green = np.array([40, 40, 180])
    upper_green = np.array([90, 255, 255])

    # Yellow
    lower_yellow = np.array([15, 70, 180])
    upper_yellow = np.array([35, 255, 255])

    # 3. Create masks
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 4. Morphological operations
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    masks = {'red': red_mask, 'green': green_mask, 'yellow': yellow_mask}
    brightness = {}
    bboxes = {}

    v_channel = hsv[:, :, 2]  # V = brightness channel

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area:
                # Make a mask for the largest contour
                contour_mask = np.zeros_like(mask)
                cv2.drawContours(contour_mask, [largest], -1, 255, thickness=cv2.FILLED)

                # Calculate brightness inside the contour
                total_brightness = np.sum(v_channel[contour_mask > 0])
                brightness[color] = total_brightness

                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest)
                bboxes[color] = ((x, y), (x + w, y + h))

    # 5. Decision: pick the color with highest brightness
    if brightness:
        detected_color = max(brightness, key=brightness.get)
        return detected_color, bboxes[detected_color]
    else:
        return None, None
