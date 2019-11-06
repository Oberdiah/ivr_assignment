#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import pyximport; pyximport.install()
import cython_functions

class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)

        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)

        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    # Recieve data and save it for camera 1's callback.
    def callback2(self, data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        size = (400, 400)
        centre1 = (400, 500)
        centre2 = (550, 500)

        img1 = self.cv_image1[
               centre1[1] - size[1] / 2:centre1[1] + size[1] / 2,
               centre1[0] - size[0] / 2:centre1[0] + size[0] / 2
               ]
        img2 = self.cv_image2[
               centre2[1] - size[1] / 2:centre2[1] + size[1] / 2,
               centre2[0] - size[0] / 2:centre2[0] + size[0] / 2
               ]

        testimg = np.asarray(cython_functions.remove_greyscale(img1))

        cv2.imshow('window1', testimg)
        cv2.imshow('window2', img2)

        cv2.waitKey(1)
        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(img2, "bgr8"))
        except CvBridgeError as e:
            print(e)


# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
