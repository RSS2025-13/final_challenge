import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_hough(img, template=None):
	"""
	Detects the two most probable left and right lane lines using color segmentation and Hough Transform.
	Returns: [((xmin, ymin), (xmax, ymax)), ((xmin, ymin), (xmax, ymax))] for left and right lines
	"""

	# Convert to HSV
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	height, width = hsv_img.shape[:2]

	# Focus on a region of interest (2nd quarter from bottom)
	portion_height = int(height * 0.43)
	start_row = portion_height
	end_row = height 
	mask_line = np.zeros(img.shape[:2], np.uint8)
	mask_line[start_row:end_row, :] = 255
	masked_image = cv2.bitwise_and(hsv_img, hsv_img, mask=mask_line)

	# cv2.imshow("Centerline Visualization", masked_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# Color threshold for white
	lower_white_bound = np.array([0, 0, 150])
	upper_white_bound = np.array([180, 50, 255])
	white_color_mask = cv2.inRange(masked_image, lower_white_bound, upper_white_bound)

	# Optional: blur slightly to clean up noise
	blurred = cv2.GaussianBlur(white_color_mask, (5, 5), 0)

	# cv2.imshow("Centerline Visualization", blurred)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# Detect lines using Probabilistic Hough Transform --- ADJUST EXPERIMENTALLY
	lines = cv2.HoughLinesP(
		blurred,
		rho=1,
		theta=np.pi / 180,
		threshold=50,
		minLineLength=30,
		maxLineGap=50
	)

	# for line in lines:
	# 	x1, y1, x2, y2 = line[0]
	# 	cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

	# cv2.imshow("Hough Lines", img)
	# cv2.waitKey(0)


	if lines is None:
		return []  # No lines detected

	# Separate lines into left and right based on slope
	left_lines = []
	right_lines = []

	for line in lines:
		x1, y1, x2, y2 = line[0]
		if x2 - x1 == 0:
			continue  # avoid division by zero
		slope = (y2 - y1) / (x2 - x1)
		if slope < -1:
			left_lines.append(((x1, y1), (x2, y2)))
		elif slope > 1:
			right_lines.append(((x1, y1), (x2, y2)))

	# for line in right_lines:
	# 	x1, y1 = line[0]
	# 	x2, y2 = line[1]
	# 	cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

	# cv2.imshow("Hough Lines", img)
	# cv2.waitKey(0)

	# Helper function to choose the most representative line
	def choose_best_line(lines, side="left"):
		if not lines:
			return None

		# Prioritize long lines with more vertical span (y1 - y2)
		def line_score(line):
			(x1, y1), (x2, y2) = line
			length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
			vertical_span = abs(y2 - y1)
			return length + 0.5 * vertical_span  # weighted score

		# Filter lines to remove very short ones (optional)
		lines = [line for line in lines if np.linalg.norm(np.subtract(line[1], line[0])) > 100]

		# Sort by score descending
		lines = sorted(lines, key=line_score, reverse=True)

		return lines[0] if lines else None


	best_left = choose_best_line(left_lines, side="left")
	best_right = choose_best_line(right_lines, side="right")

	if best_left is None or best_right is None:
		return []  # Didn't find both lines
	

	return [best_left, best_right]

