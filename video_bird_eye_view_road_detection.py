import cv2
# import pafy
import numpy as np
from line_fit import line_fit, calc_curve, calc_vehicle_offset
from Line import Line

from hybridnets import HybridNets, optimized_model
from hybridnets.utils import get_horizon_points


# TODO: If you use a different video, modify the horizon points. 
# To get new horizon points, you can set this variable to None and 
# the horizon point selection function will run automatically. Copy the 
# printed line below once you get the points
# horizon_points = np.array([[605., 464.],
#        					   [827., 475.]], dtype=np.float16)
horizon_points = None

# Initialize video
cap = cv2.VideoCapture("challenge_video.mp4")

# videoUrl = 'https://youtu.be/jvRDlJvG8E8'
# videoPafy = pafy.new(videoUrl)
# print(videoPafy.streams)
# cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 0 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize road detector
model_path = "models/hybridnets_512x640.onnx"
anchor_path = "models/anchors_512x640.npy"
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280+720,720))

cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

	try:
		# Read frame from the video
		ret, new_frame = cap.read()
		if not ret:	
			break

		if horizon_points is None:
			horizon_points = get_horizon_points(new_frame)

	except:
		continue
	
	# Update road detector
	seg_map, filtered_boxes, filtered_scores = roadEstimator(new_frame)
	blank_image = np.zeros((new_frame.shape[1], new_frame.shape[0], 3), np.uint8)


		# combined_img = roadEstimator.draw_all(new_frame, horizon_points)
	combined_img = roadEstimator.draw_bird_eye(blank_image, horizon_points)
	# combined_img = roadEstimator.draw_perspect(blank_image)

	def calculate_lane_curvature(bird_eye_seg_map, ploty):
		# Identify lane pixels
		nonzero = bird_eye_seg_map.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		lane_inds = ((nonzerox > 0) & (nonzerox < bird_eye_seg_map.shape[1])).nonzero()[0]

		# Fit a polynomial to the lane pixels
		leftx = nonzerox[lane_inds]
		lefty = nonzeroy[lane_inds]
		left_fit = np.polyfit(lefty, leftx, 2)

		# Calculate curvature
		y_eval = np.max(ploty)

		# y_eval = 700 * ym
		left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])

		return left_curvature

	# Generate some sample data for ploty
	ploty = np.linspace(0, combined_img.shape[0]-1, combined_img.shape[0])

	# Calculate lane curvature
	lane_curvature = calculate_lane_curvature(combined_img, ploty)
	print("Lane Curvature:", lane_curvature)
		
	# 	# ym = 30 / 720
	# 	# xm = 3.7 / 700


	cv2.imshow("Road Detections", combined_img)
	out.write(combined_img)

out.release()