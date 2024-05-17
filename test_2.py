import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from hybridnets import HybridNets, optimized_model
from hybridnets.utils import get_horizon_points
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset


# Initialize video
cap = cv2.VideoCapture("challenge_video.mp4")
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)


# Initialize road detector
model_path = "models/hybridnets_512x640.onnx"
anchor_path = "models/anchors_512x640.npy"
optimized_model(model_path)  # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video writer
out = cv2.VideoWriter('output_curv.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))

# Main loop for processing frames
while cap.isOpened():

    if cv2.waitKey(1) == ord('q'):
        break
    # Read frame from the video


    try:
        ret, new_frame = cap.read()
        if not ret:
            break
        seg_map, filtered_boxes, filtered_scores = roadEstimator(new_frame)
        
        window_size = 5  # how many frames for line smoothing
        left_line = Line(n=window_size)
        right_line = Line(n=window_size)
        detected = False  # did the fast line fit detect the lines?
        left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
        left_lane_inds, right_lane_inds = None, None  # for calculating curvature
        

        def estimate_curvature(new_frame):
            blank_image = np.zeros((new_frame.shape[1], new_frame.shape[0], 3), np.uint8)
            bird_eye, m_inv = roadEstimator.draw_perspect(blank_image)
            combined_img = roadEstimator.draw_segmentation(blank_image)
            detected = False
            if not detected:
                ret = line_fit(bird_eye)
                left_fit = ret['left_fit']
                right_fit = ret['right_fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                left_lane_inds = ret['left_lane_inds']
                right_lane_inds = ret['right_lane_inds']
                
                # Get moving average of line fit coefficients
                left_fit = left_line.add_fit(left_fit)
                right_fit = right_line.add_fit(right_fit)
                
                # Calculate curvature

                left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
                detected = True  # slow line fit always detects the line
            else:  # implies detected == True
                left_fit = left_line.get_fit()
                right_fit = right_line.get_fit()
                ret = tune_fit(bird_eye, left_fit, right_fit)
                left_fit = ret['left_fit']
                right_fit = ret['right_fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                left_lane_inds = ret['left_lane_inds']
                right_lane_inds = ret['right_lane_inds']
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = left_line.add_fit(left_fit)
                    right_fit = right_line.add_fit(right_fit)
                    left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
                else:
                    detected = False

            vehicle_offset = calc_vehicle_offset(combined_img, left_fit, right_fit)

			# Perform final visualization on top of original undistorted image
            result = final_viz(combined_img, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

            return result


        lane_detection = estimate_curvature(new_frame)
        cv2.imshow("Lane Lines", lane_detection)
        

    except Exception as e:
        print("Error processing frame:", e)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
