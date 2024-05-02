import cv2
import numpy as np
from hybridnets import HybridNets, optimized_model
from hybridnets.utils import get_horizon_points


horizon_points = np.array([[605., 464.],
       					   [827., 475.]], dtype=np.float16)
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
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280 + 720, 720))

# Main loop for processing frames
while cap.isOpened():
    # Read frame from the video
    ret, new_frame = cap.read()
    if not ret:
        break

    try:
        # Update road detector
        seg_map, filtered_boxes, filtered_scores = roadEstimator(new_frame)
        blank_image = np.zeros((new_frame.shape[1], new_frame.shape[0], 3), np.uint8)
        combined_img = roadEstimator.draw_bird_eye(blank_image, horizon_points)
        combined_seg = roadEstimator.draw_segmentation(new_frame)

        # Calculate curvature
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
            curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])

            return curvature

        # Generate some sample data for ploty
        ploty = np.linspace(0, combined_img.shape[0] - 1, combined_img.shape[0])

        # Calculate lane curvature
        lane_curvature = calculate_lane_curvature(combined_img, ploty)

        # Draw curvature information on the frame
        cv2.putText(combined_seg, "Lane Curvature: {:.2f}".format(lane_curvature), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame with curvature information
        cv2.imshow("Road Detections", combined_seg)
        out.write(combined_seg)

    except Exception as e:
        print("Error processing frame:", e)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
