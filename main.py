#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import cv_bridge
import time

class ImageCollagePublisher:
    def __init__(self, topics, output_topic, timeout=.000001):
        rospy.init_node('image_collage_publisher', anonymous=True)
        self.bridge = cv_bridge.CvBridge()
        self.images = [None] * len(topics)
        self.last_received = [time.time()] * len(topics)
        self.timeout = timeout
        self.publisher = rospy.Publisher(output_topic, CompressedImage, queue_size=1)
        self.subscribers = [rospy.Subscriber(topic, CompressedImage, self.callback, callback_args=index) for index, topic in enumerate(topics)]
        self.check_timer = rospy.Timer(rospy.Duration(1), self.check_stale_images)

    def callback(self, data, index):
        try:
            if data is not CompressedImage:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            else:
                rospy.logerr("Received empty image.")
                height, width = 1080, 1920  # Assuming initial image size
                cv_image = np.zeros((height, width, 3), dtype=np.uint8)

            cv_image = cv2.resize(cv_image, None, fx=0.1667, fy=0.1667, interpolation=cv2.INTER_AREA)
            self.images[index] = cv_image
            self.last_received[index] = time.time()  # Update the last received time

            if all(image is not None for image in self.images):  
                rospy.loginfo("All images received, attempting to publish collage.")
                self.publish_collage()
        except cv_bridge.CvBridgeError as e:
            rospy.logerr("Could not convert image: %s" % str(e))
            self.images[index] = np.zeros((180, 320, 3), dtype=np.uint8)  # Adjusted size to 1/6 of original

    def check_stale_images(self, event):
        current_time = time.time()
        for i, last_time in enumerate(self.last_received):
            if current_time - last_time > self.timeout:
                # Image is stale, replace with black image
                self.images[i] = np.zeros((180, 320, 3), dtype=np.uint8)  # Adjusted size to 1/6 of original
                rospy.logwarn("No new image from topic index %d, using black image." % i)

    def publish_collage(self):
        if all(image is not None for image in self.images):  
            top = np.hstack((self.images[0], self.images[1], self.images[2]))
            bottom = np.hstack((self.images[0], self.images[3], self.images[2]))
            collage = np.vstack((top, bottom))

            try:
                msg = self.bridge.cv2_to_compressed_imgmsg(collage)
                self.publisher.publish(msg)
                rospy.loginfo("Collage published.")
            except cv_bridge.CvBridgeError as e:
                rospy.logerr("Could not convert collage to compressed message: %s" % str(e))
            finally:
                self.images = [None] * len(self.images)
        else:
            rospy.loginfo("Not all images are ready for collage creation.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    topics = [
        "/clpe_ros/cams_1/image_raw/compressed",
        "/clpe_ros/cams_0/image_raw/compressed",
        "/clpe_ros/cams_2/image_raw/compressed",
        "/clpe_ros/cams_3/image_raw/compressed"
    ]
    output_topic = "/image_collage/compressed"
    collage_publisher = ImageCollagePublisher(topics, output_topic)
    collage_publisher.run()
