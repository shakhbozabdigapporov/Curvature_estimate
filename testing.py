import cv2
import matplotlib.pyplot as plt
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

out = cv2.VideoWriter('output_curv_demo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_curv_demo.avi', fourcc, 20.0, (1280, 720))

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
        binary_warped, unwarped = roadEstimator.draw_perspect(new_frame)

        def find_lane_lines(binary_warped):
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = int(histogram.shape[0] // 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # HYPERPARAMETERS
            # Choose the number of sliding windows
            nwindows = 9
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50

            # Set height of windows - based on nwindows above and image shape
            window_height = int(binary_warped.shape[0] // nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Current positions to be updated later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # If found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img
        
        def calculate_curvature(binary_warped, left_fit, right_fit):
            # Define y-value where we want radius of curvature
            # Choose the maximum y-value, corresponding to the bottom of the image
            y_eval = binary_warped.shape[0]

            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 30 / 720 # meters per pixel in y dimension
            xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(np.array(range(binary_warped.shape[0])) * ym_per_pix, left_fit[0] * (np.array(range(binary_warped.shape[0])) ** 2) * xm_per_pix +
                                    left_fit[1] * np.array(range(binary_warped.shape[0])) * xm_per_pix + left_fit[2], 2)
            right_fit_cr = np.polyfit(np.array(range(binary_warped.shape[0])) * ym_per_pix, right_fit[0] * (np.array(range(binary_warped.shape[0])) ** 2) * xm_per_pix +
                                    right_fit[1] * np.array(range(binary_warped.shape[0])) * xm_per_pix + right_fit[2], 2)

            # Calculate the new radii of curvature
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])

            return left_curverad, right_curverad
        
        left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = find_lane_lines(binary_warped)
        left_curverad, right_curverad = calculate_curvature(binary_warped, left_fit, right_fit)

        print(f"Left curvature: {left_curverad} m")
        print(f"Right curvature: {right_curverad} m")
        plt.imshow(out_img)

        plt.plot(left_fit[0]*np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])**2 + left_fit[1]*np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]) + left_fit[2],
                 np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]), color='yellow')
        plt.plot(right_fit[0]*np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])**2 + right_fit[1]*np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]) + right_fit[2],
                np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]), color='yellow')
        plt.show()
        # # Display frame with bird_eye_view information
        # cv2.imshow("Road Detections", unwarped)
        # out.write(unwarped)

    except Exception as e:
        print("Error processing frame:", e)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
