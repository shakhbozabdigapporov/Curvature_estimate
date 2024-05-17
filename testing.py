import cv2
import numpy as np
from hybridnets import HybridNets, optimized_model
from hybridnets.utils import get_horizon_points




# horizon_points = np.array([[605., 464.],
#        					   [827., 475.]], dtype=np.float16)
horizon_points = None
# Initialize video
cap = cv2.VideoCapture("rec_ros.mp4")
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

print('horizon_points',horizon_points)

# Initialize road detector
model_path = "models/hybridnets_512x640.onnx"
anchor_path = "models/anchors_512x640.npy"
optimized_model(model_path)  # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video writer

# out = cv2.VideoWriter('output_curv_demo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_curv_demo.avi', fourcc, 20.0, (1280, 720))

# Main loop for processing frames
while cap.isOpened():

    if cv2.waitKey(1) == ord('q'):
        break
    # Read frame from the video


    try:
        ret, new_frame = cap.read()
        if not ret:
            break
        # if horizon_points is None:
        #     horizon_points = get_horizon_points(new_frame)
        # # Update road detector
        
        seg_map, filtered_boxes, filtered_scores = roadEstimator(new_frame)
        blank_image = np.zeros((new_frame.shape[1], new_frame.shape[0], 3), np.uint8)
        combined_img = roadEstimator.draw_segmentation(new_frame)
        bird_eye, unwarped = roadEstimator.draw_perspect(blank_image)
        # combined_seg = roadEstimator.draw_segmentation(new_frame)

        # Calculate curvature with this code can be calculated just uncomment the lines 
        # begin with y_eval
        def curvature_estimate(bird_eye_seg_map, ploty):
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

            return left_fit, curvature


        # Generate some sample data for ploty
        # ploty = np.linspace(0, bird_eye.shape[0] - 1, bird_eye.shape[0])
        ploty = np.linspace(0, combined_img.shape[0] - 1, combined_img.shape[0])

        def draw_lane_lines(bird_eye_seg_map, left_fit, ploty):
        # Generate points along the curve
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        # Create a blank canvas to draw the lane lines
            canvas = np.zeros_like(bird_eye_seg_map)

        # Draw the left lane line
            for i in range(len(left_fitx) - 1):
                cv2.line(canvas, (int(left_fitx[i]), int(ploty[i])), 
                        (int(left_fitx[i+1]), int(ploty[i+1])), (0, 255, 0), thickness=5)

            return canvas
        

        poly_test, lane_curvature = curvature_estimate(bird_eye, ploty)

        lane_lines_canvas = draw_lane_lines(bird_eye, poly_test, ploty)
        # cv2.imshow("Lane Lines", lane_lines_canvas)
        

        # # Draw curvature information on the frame
        cv2.putText(combined_img, "Lane Curvature: {:.2f}".format(lane_curvature), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow("lane Curvature", combined_seg)
    
        # Display frame with bird_eye_view information
        cv2.imshow("Road Detections", combined_img)
        out.write(combined_img)

    except Exception as e:
        print("Error processing frame:", e)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
