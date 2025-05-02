import cv2
import numpy as np
from computer_vision.color_hough_segmentation import cd_color_hough

def visualize_centerline(image_path):
    """
    Loads an image, computes lane lines using cd_color_hough, computes and overlays the centerline.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image:", image_path)
        return

    lines = cd_color_hough(image)

    if not lines or len(lines) != 2:
        print("Could not detect both lane lines.")
        return

    left_line, right_line = lines
    (lx1, ly1), (lx2, ly2) = left_line
    (rx1, ry1), (rx2, ry2) = right_line

    # Fit lines
    left_m = (ly2 - ly1) / (lx2 - lx1 + 1e-6)
    left_b = ly1 - left_m * lx1

    right_m = (ry2 - ry1) / (rx2 - rx1 + 1e-6)
    right_b = ry1 - right_m * rx1

    num_points = 10

    min_y = int(max(min(ly1, ly2), min(ry1, ry2)))
    max_y = int(min(max(ly1, ly2), max(ry1, ry2)))

    ys = np.linspace(min_y, max_y, num_points)
    center_points = []

    # lxs = np.linspace(lx1, lx2, num_points)
    # lys = np.linspace(ly1, ly2, num_points)

    # rxs = np.linspace(rx1, rx2, num_points)
    # rys = np.linspace(ry1, ry2, num_points)

    # for i in range(num_points):
    #     cpx = int((lxs[i] + rxs[i]) * 0.5)
    #     cpy = int((lys[i] + rys[i]) * 0.5)
    #     center_points.append((cpx, cpy))


    for y in ys:
        lx = (y - left_b) / (left_m + 1e-6)
        rx = (y - right_b) / (right_m + 1e-6)
        cx = int((lx + rx) / 2.0)
        cy = int(y)
        center_points.append((cx, cy))
    
    print(center_points)

    # Draw the lines and centerline
    output = image.copy()
    cv2.line(output, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)  # Left line (green)
    cv2.line(output, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)  # Right line (blue)

    for pt in center_points:
        cv2.circle(output, pt, 3, (0, 0, 255), -1)  # Red centerline

    cv2.imshow("Centerline Visualization", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


visualize_centerline("racetrack_images/lane_3/image3.png")
