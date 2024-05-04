import cv2
# import pafy
import numpy as np
# from line_fit import line_fit, calc_curve, calc_vehicle_offset
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

out = cv2.VideoWriter('output_bird_eye.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720))

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

		# if horizon_points is None:
		# 	horizon_points = get_horizon_points(new_frame)

	except:
		continue
	print("horizon_points", horizon_points)
	# Update road detector
	seg_map, filtered_boxes, filtered_scores = roadEstimator(new_frame)
	blank_image = np.zeros((new_frame.shape[1], new_frame.shape[0], 3), np.uint8)


	combined_img = roadEstimator.draw_perspect(blank_image)

	# combined_img = roadEstimator.draw_perspect(blank_image)
 
		
	# 	# ym = 30 / 720
	# 	# xm = 3.7 / 700


	cv2.imshow("Road Detections", combined_img)
	out.write(combined_img)

out.release()