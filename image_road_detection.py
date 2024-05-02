import cv2
import numpy as np
# from imread_from_url import imread_from_url

from hybridnets import HybridNets, optimized_model
horizon_points = np.array([[500., 360.],
       					   [700., 360.]], dtype=np.float16)

model_path = "models/hybridnets_512x640.onnx"
anchor_path = "models/anchors_512x640.npy"

# Initialize road detector
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

img = cv2.imread('highway.jpg')

# Update road detector
seg_map, filtered_boxes, filtered_scores = roadEstimator(img)

combined_img = roadEstimator.draw_2D(img)
# combined_img = roadEstimator.draw_bird_eye(img, horizon_points)
cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Road Detections", combined_img)

cv2.imwrite("output.jpg", combined_img)
cv2.waitKey(0)