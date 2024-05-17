import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip
from hybridnets import HybridNets, optimized_model
from hybridnets.utils import get_horizon_points
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset


# horizon_points = np.array([[605., 464.],
#        					   [827., 475.]], dtype=np.float16)
horizon_points = None
# Initialize video
cap = cv2.VideoCapture("challenge_video.mp4")
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

print('horizon_points',horizon_points)

# Initialize road detector
model_path = "models/hybridnets_512x640.onnx"
anchor_path = "models/anchors_512x640.npy"
optimized_model(model_path)  # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video writer
out = cv2.VideoWriter('output_curv.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))
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
        # Update road detector
        seg_map, filtered_boxes, filtered_scores = roadEstimator(new_frame)
        blank_image = np.zeros((new_frame.shape[1], new_frame.shape[0], 3), np.uint8)
        combined_img = roadEstimator.draw_segmentation(blank_image)
        bird_eye = roadEstimator.draw_perspect(blank_image)
        combined_seg = roadEstimator.draw_segmentation(new_frame)




		window_size = 5  # how many frames for line smoothing
		left_line = Line(n=window_size)
		right_line = Line(n=window_size)
		detected = False  # did the fast line fit detect the lines?
		left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
		left_lane_inds, right_lane_inds = None, None  # for calculating curvature


# MoviePy video annotation will call this function
		def estimate_curvature(combined_img):

			# Perform polynomial fit
			if not detected:
				# Slow line fit
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
				# Fast line fit
				left_fit = left_line.get_fit()
				right_fit = right_line.get_fit()
				ret = tune_fit(binary_warped, left_fit, right_fit)
				left_fit = ret['left_fit']
				right_fit = ret['right_fit']
				nonzerox = ret['nonzerox']
				nonzeroy = ret['nonzeroy']
				left_lane_inds = ret['left_lane_inds']
				right_lane_inds = ret['right_lane_inds']

				# Only make updates if we detected lines in current frame
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

			vehicle_offset = calc_vehicle_offset(seg_map, left_fit, right_fit)

			# Perform final visualization on top of original undistorted image
			result = final_viz(seg_map, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

			return result


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	# Annotate the video
	annotate_video('project_video.mp4', 'out.mp4')

	# Show example annotated image on screen for sanity check
	img_file = 'test_images/test2.jpg'
	img = mpimg.imread(img_file)
	result = annotate_image(img)
	result = annotate_image(img)
	result = annotate_image(img)
	plt.imshow(result)
	plt.show()
