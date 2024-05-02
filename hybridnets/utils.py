import numpy as np
import cv2

segmentation_colors = np.array([[0,    0,    0],
								[0,  0,  0],
								[255,  255,   255]], dtype=np.uint8)

print("seg_colors", segmentation_colors)

detection_color = (191,  255,  0)
label = "car"

ORIGINAL_HORIZON_POINTS = np.float32([[571, 337], [652, 337]])

num_horizon_points = 0
new_horizon_points = []
def util_draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors

	color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))

	color_segmap[seg_map>0] = segmentation_colors[seg_map[seg_map>0]]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))
	# print("segmask: ", seg_map.shape)


	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)
	
	


	# # Fuse both images
	# if(alpha == 0):
	# 	combined_img = np.hstack((image, color_segmap))
	# else:
	# 	combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)
		# combined_img = plot(combined_img)
	

	return color_segmap


def measure_curvature(self):
        """
        Measure the curvature of the lane lines.

        Returns:
            Tuple: Left lane curvature, right lane curvature, and position of the vehicle from the lane center.
        """
        ym = 30 / 720
        xm = 3.7 / 700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(self.left_fit, [700 ** 2, 700, 1])
        xr = np.dot(self.right_fit, [700 ** 2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm
        return left_curveR, right_curveR, pos

def plot(self, out_img):
	"""
	Plot lane lines and text overlays on the image.

	Args:
		out_img (np.array): Image with polynomial curves plotted on it.

	Returns:
		np.array: Image with lane lines, text overlays, and direction indicators.
	"""
	np.set_printoptions(precision=6, suppress=True)
	lR, rR, pos = self.measure_curvature()

	value = None
	if abs(self.left_fit[0]) > abs(self.right_fit[0]):
		value = self.left_fit[0]
	else:
		value = self.right_fit[0]

	if abs(value) <= 0.00015:
		self.dir.append('F')
	elif value < 0:
		self.dir.append('L')
	else:
		self.dir.append('R')

	if len(self.dir) > 10:
		self.dir.pop(0)

	W = 400
	H = 430
	widget = np.copy(out_img[:H, :W])
	widget //= 2
	widget[0, :] = [0, 0, 255]
	widget[-1, :] = [0, 0, 255]
	widget[:, 0] = [0, 0, 255]
	widget[:, -1] = [0, 0, 255]
	out_img[:H, :W] = widget

	direction = max(set(self.dir), key=self.dir.count)
	msg = "LKAS: Keep Straight Ahead"
	curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
	if direction == 'L':
		y, x = self.left_curve_img[:, :, 3].nonzero()
		out_img[y, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
		msg = "LKAS: Left Curve Ahead"
	if direction == 'R':
		y, x = self.right_curve_img[:, :, 3].nonzero()
		out_img[y, x - 100 + W // 2] = self.right_curve_img[y, x, :3]
		msg = "LKAS: Right Curve Ahead"
	if direction == 'F':
		y, x = self.keep_straight_img[:, :, 3].nonzero()
		out_img[y, x - 100 + W // 2] = self.keep_straight_img[y, x, :3]

	cv2.putText(out_img, msg, org=(40, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
	if direction in 'LR':
		cv2.putText(out_img, curvature_msg, org=(40, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)

	cv2.putText(out_img, "LDWS: Good Lane Keeping", org=(10, 350), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

	cv2.putText(out_img, "Vehicle is {:.2f}m away from center".format(pos), org=(10, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66, color=(255, 255, 255), thickness=2)

	return out_img


# Ref: https://github.com/datvuthanh/HybridNets/blob/d43b0aa8de2a1d3280084270d29cf4c7abf640ae/utils/plot.py#L52


# def util_draw_detections(boxes, scores, image, text=True):

# 	tl = int(round(0.0015 * max(image.shape[0:2])))  # line thickness
# 	tf = max(tl, 1)  # font thickness
# 	for box, score in zip(boxes, scores):
# 		c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
# 		cv2.rectangle(image, c1, c2, detection_color, thickness=tl)
# 		if text:
# 			s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
# 			t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
# 			c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
# 			cv2.rectangle(image, c1, c2, detection_color, -1)  # filled
# 			cv2.putText(image, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
# 						thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

# 	return image

def util_draw_bird_eye_view(seg_map, hoizon_points=ORIGINAL_HORIZON_POINTS):

	img_h, img_w = seg_map.shape[:2]
	bird_eye_view_w, bird_eye_view_h = (img_w, img_h)
	offset = bird_eye_view_w/2.5
	# offset = bird_eye_view_w/1.3
	bird_eye_view_points = np.float32([[offset, bird_eye_view_h], [bird_eye_view_w - offset, bird_eye_view_h], 
										[offset, 0], [bird_eye_view_w - offset, 0]])

	image_points = np.vstack((np.float32([[0, img_h], [img_w, img_h]]), hoizon_points))
	M = cv2.getPerspectiveTransform(image_points, bird_eye_view_points)
	bird_eye_seg_map = cv2.warpPerspective(seg_map, M, (bird_eye_view_w, bird_eye_view_h))
	return bird_eye_seg_map


def perspective_transform(seg_map):
	img_size = (seg_map.shape[1], seg_map.shape[0])
	src = np.float32(
		[[200, 720],
		[1100, 720],
		[595, 450],
		[685, 450]])
	dst = np.float32(
		[[300, 720],
		[980, 720],
		[300, 0],
		[980, 0]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(seg_map, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

	return warped





# def calculate_lane_curvature(bird_eye_seg_map, ploty):
#     # Identify lane pixels
#     nonzero = bird_eye_seg_map.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     lane_inds = ((nonzerox > 0) & (nonzerox < bird_eye_seg_map.shape[1])).nonzero()[0]

#     # Fit a polynomial to the lane pixels
#     leftx = nonzerox[lane_inds]
#     lefty = nonzeroy[lane_inds]
#     left_fit = np.polyfit(lefty, leftx, 2)

#     # Calculate curvature
#     y_eval = np.max(ploty)
#     left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])

#     return left_curvature





def util_draw_aerial_view(seg_map, hoizon_points=ORIGINAL_HORIZON_POINTS):
    img_h, img_w = seg_map.shape[:2]
    aerial_view_w, aerial_view_h = (img_w, img_h)
    offset = aerial_view_w / 2.5  # Adjust the offset as needed

    # Define the points for the aerial view (looking directly down onto the scene)
    aerial_view_points = np.float32([[0, aerial_view_h], 
                                     [aerial_view_w, aerial_view_h], 
                                     [0, 0], 
                                     [aerial_view_w, 0]])

    # Define the image points (including the horizon)
    image_points = np.vstack((np.float32([[0, img_h], [img_w, img_h]]), hoizon_points))

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(image_points, aerial_view_points)

    # Apply the perspective transformation to the segmentation map
    aerial_seg_map = cv2.warpPerspective(seg_map, M, (aerial_view_w, aerial_view_h))

    return aerial_seg_map





# Ref: https://github.com/datvuthanh/HybridNets/blob/d43b0aa8de2a1d3280084270d29cf4c7abf640ae/utils/utils.py#L615
def transform_boxes(boxes, anchors):

	y_centers_a = (anchors[:, 0] + anchors[:, 2]) / 2
	x_centers_a = (anchors[:, 1] + anchors[:, 3]) / 2
	ha = anchors[:, 2] - anchors[:, 0]
	wa = anchors[:, 3] - anchors[:, 1]

	w = np.exp(boxes[:, 3]) * wa
	h = np.exp(boxes[:, 2]) * ha

	y_centers = boxes[:, 0] * ha + y_centers_a
	x_centers = boxes[:, 1] * wa + x_centers_a

	ymin = y_centers - h / 2.
	xmin = x_centers - w / 2.
	ymax = y_centers + h / 2.
	xmax = x_centers + w / 2.

	return np.vstack((xmin, ymin, xmax, ymax)).T


# Ref: https://python-ai-learn.com/2021/02/14/nmsfast/
def iou_np(box, boxes, area, areas):

	x_min = np.maximum(box[0], boxes[:,0])
	y_min = np.maximum(box[1], boxes[:,1])
	x_max = np.minimum(box[2], boxes[:,2])
	y_max = np.minimum(box[3], boxes[:,3])

	w = np.maximum(0, x_max - x_min + 1)
	h = np.maximum(0, y_max - y_min + 1)
	intersect = w*h
	
	iou_np = intersect / (area + areas - intersect)
	return iou_np

# Ref: https://python-ai-learn.com/2021/02/14/nmsfast/
def nms_fast(bboxes, scores, iou_threshold=0.5):
	 
	areas = (bboxes[:,2] - bboxes[:,0] + 1) \
			 * (bboxes[:,3] - bboxes[:,1] + 1)
	
	sort_index = np.argsort(scores)
	
	i = -1
	while(len(sort_index) >= 1 - i):

		max_scr_ind = sort_index[i]
		ind_list = sort_index[:i]

		iou = iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
					 areas[max_scr_ind], areas[ind_list])
		
		del_index = np.where(iou >= iou_threshold)
		sort_index = np.delete(sort_index, del_index)
		i -= 1
	
	bboxes = bboxes[sort_index]
	scores = scores[sort_index]
	
	return bboxes, scores

def get_horizon_points(image):

	cv2.namedWindow("Get horizon points", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("Get horizon points", get_horizon_point)

	# Draw horizontal line
	image = cv2.line(image, (0,image.shape[0]//2), 
							(image.shape[1],image.shape[0]//2), 
							(0,  0,   251), 1)

	cv2.imshow("Get horizon points", image)

	num_lines = 0
	while True:

		if (num_lines == 0) and (num_horizon_points == 1):

			image = cv2.line(image, (0,image.shape[0]), 
							(new_horizon_points[0][0], new_horizon_points[0][1]), 
							(192,  67,   251), 3)

			image = cv2.circle(image, (new_horizon_points[0][0], new_horizon_points[0][1]), 
							5, (251,  191,   67), -1)
			
			cv2.imshow("Get horizon points", image)
			num_lines += 1

		elif(num_lines == 1) and (num_horizon_points == 2):

			image = cv2.line(image, (image.shape[1],image.shape[0]), 
				(new_horizon_points[1][0], new_horizon_points[1][1]), 
				(192,  67,   251), 3)

			image = cv2.circle(image, (new_horizon_points[1][0], new_horizon_points[1][1]), 
								5, (251,  191,   67), -1)
			
			cv2.imshow("Get horizon points", image)
			num_lines += 1
			break

		cv2.waitKey(100)

	cv2.waitKey(1000)
	cv2.destroyWindow("Get horizon points")

	horizon_points = np.float32(new_horizon_points)
	print(f"horizon_points = np.{repr(horizon_points)}")

	return horizon_points

def get_horizon_point(event,x,y,flags,param):

	global num_horizon_points, new_horizon_points 

	if event == cv2.EVENT_LBUTTONDBLCLK:

		new_horizon_points.append([x,y])
		num_horizon_points += 1
