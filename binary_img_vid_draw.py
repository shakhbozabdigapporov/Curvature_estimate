import cv2
import matplotlib.pyplot as plt
import numpy as np
from hybridnets import HybridNets, optimized_model
from hybridnets.utils import get_horizon_points


# horizon_points = np.array([[605., 464.],
#        					   [827., 475.]], dtype=np.float16)
horizon_points = None
# Initialize video
cap = cv2.VideoCapture("output3.mp4")
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
        # combined_img = roadEstimator.draw_segmentation(new_frame)
        binary_warped, unwarped = roadEstimator.draw_perspect(blank_image)

        def find_lane_lines(binary_warped):
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            # out_img = binary_warped * 255

            
            # Find the peak of the left and right halves of the histogram
            midpoint = int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[100:midpoint])+100
            rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
                ## Plot the histogram
            # plt.plot(histogram)
            # plt.title('Histogram of Lane Line Pixels')
            # plt.xlabel('Pixel Position')
            # plt.ylabel('Frequency')

            # # Plot the identified left and right base positions
            # plt.axvline(x=leftx_base, color='r', linestyle='--', label='Left Lane Base')
            # plt.axvline(x=rightx_base, color='g', linestyle='--', label='Right Lane Base')
            # plt.axvline(x=midpoint, color='orange', linestyle='--', label = 'midpoint')

            # # Add legend
            # plt.legend()

            # # Show the plot
            # plt.show()

            # HYPERPARAMETERS
            nwindows = 13
            margin = 230
            minpix = 30

            window_height = int(binary_warped.shape[0] // nwindows)
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx_current = leftx_base
            rightx_current = rightx_base

            left_lane_inds = []
            right_lane_inds = []

            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Ensure coordinates are integers and within valid range
                win_y_low = np.clip(win_y_low, 0, binary_warped.shape[0])
                win_y_high = np.clip(win_y_high, 0, binary_warped.shape[0])
                win_xleft_low = np.clip(win_xleft_low, 0, binary_warped.shape[1])
                win_xleft_high = np.clip(win_xleft_high, 0, binary_warped.shape[1])
                win_xright_low = np.clip(win_xright_low, 0, binary_warped.shape[1])
                win_xright_high = np.clip(win_xright_high, 0, binary_warped.shape[1])

                cv2.rectangle(out_img, (int(win_xleft_low), int(win_y_low)), (int(win_xleft_high), int(win_y_high)), (0, 255, 0), 2)
                # cv2.rectangle(out_img, (int(win_xright_low), int(win_y_low)), (int(win_xright_high), int(win_y_high)), (0, 255, 0), 2)

                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # left_fit = np.polyfit(lefty, leftx, 2)
            # right_fit = np.polyfit(righty, rightx, 2)
            if len(leftx) > 0 and len(lefty) > 0:
                left_fit = np.polyfit(lefty, leftx, 2)
            else:
                left_fit = None
                print("No lane line detected on the left side")

            if len(rightx) > 0 and len(righty) > 0:
                right_fit = np.polyfit(righty, rightx, 2)
            else:
                right_fit = None
                print("No lane line detected on the right side")

            return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img
        
        def draw_lane_lines(binary_warped, left_fit, right_fit):
            # Ensure the binary_warped image is in grayscale
            if len(binary_warped.shape) == 3 and binary_warped.shape[2] == 3:
                print("Converting binary_warped to grayscale.")
                binary_warped_gray = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
            elif len(binary_warped.shape) == 2:
                print("binary_warped is already grayscale.")
                binary_warped_gray = binary_warped
            else:
                raise ValueError("binary_warped has an unexpected shape.")
            
            # Generate y values
            ploty = np.linspace(0, binary_warped_gray.shape[0] - 1, binary_warped_gray.shape[0])

            # Create an image to draw the lines on
            warp_zero = np.zeros_like(binary_warped_gray).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Draw the lane onto the warped blank image
            if left_fit is not None:
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_left = pts_left.astype(int)  # Ensure coordinates are integers
                cv2.polylines(color_warp, [pts_left], isClosed=False, color=(255, 0, 0), thickness=150)
            else:
                print("No left lane detected.")

            if right_fit is not None:
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                pts_right = pts_right.astype(int)  # Ensure coordinates are integers
                cv2.polylines(color_warp, [pts_right], isClosed=False, color=(0, 0, 255), thickness=150)
            else:
                print("No right lane detected.")

            # Combine the result with the original image
            result = cv2.addWeighted(cv2.cvtColor(binary_warped_gray, cv2.COLOR_GRAY2BGR), 1, color_warp, 0.7, 0)
            return result

        def calculate_curvature(binary_warped, left_fit, right_fit):
            y_eval = 1079

            ym_per_pix = 30 / 1080  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 1140  # meters per pixel in x dimension

        # TODO: work on drawing this ploty
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

            if left_fit is not None:
                leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
                left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * left_fit_cr[0])
            else:
                left_curverad = float('inf')

            if right_fit is not None:
                rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
                right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
                right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * right_fit_cr[0])
            else:
                right_curverad = float('inf')

            # If one side is not detected, use the curvature of the detected side
            if left_fit is None and right_fit is not None:
                left_curverad = right_curverad
            elif right_fit is None and left_fit is not None:
                right_curverad = left_curverad

            return left_curverad, right_curverad
        left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = find_lane_lines(binary_warped)
        result = draw_lane_lines(binary_warped, left_fit, right_fit)


        left_curverad, right_curverad = calculate_curvature(binary_warped, left_fit, right_fit)

        print(f"Left curvature: {left_curverad} m")
        print(f"Right curvature: {right_curverad} m")
        cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)
        # cv2.imshow("Road Detections", combined_img)
        cv2.imshow("Road Detections", result)
        # cv2.imwrite("output.jpg", combined_img)
        # cv2.waitKey(0)

    except Exception as e:
        print("Error processing frame:", e)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
