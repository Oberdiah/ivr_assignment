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

        self.target_x_pub = rospy.Publisher("target_x", Float64, queue_size=10)
        self.target_y_pub = rospy.Publisher("target_y", Float64, queue_size=10)
        self.target_z_pub = rospy.Publisher("target_z", Float64, queue_size=10)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    # Recieve data and save it for camera 1's callback.
    def callback2(self, data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def get_positions(self, img, img_num):
        img = img.copy()
        cython_functions.remove_greyscale(img)
        cython_functions.saturate(img)

        yellow = np.array([0, 255, 255])
        blue = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        red = np.array([0, 0, 255])
        orange = np.array([108, 196, 253])

        targets = [
            self.process_color(img, yellow, 5),
            self.process_color(img, blue, 5),
            self.process_color(img, green, 5),
            self.process_color(img, red, 5),
            self.process_orange(img, orange, 16)
        ]

        if img_num == 2:
            cv2.imshow('orange target', targets[4])

        positions = [0,0,0,0,0]

        for i in range(5):
            M = cv2.moments(targets[i])
            if M['m00'] == 0:
                positions[i] = (-1,-1)
                print("Warning. Sphere " + str(i) + " has been lost in image " + str(img_num))
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            positions[i] = (cx, cy)

        return positions

    def process_color(self, img, colour, colour_threshold):
        return self.greyscale(cython_functions.select_colour(img.copy(), colour, colour_threshold))

    def greyscale(self, img):
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)

    def process_orange(self, img, colour, colour_threshold):
        img = cython_functions.select_colour(img.copy(), colour, colour_threshold)
        # Erode the image enough to get rid of the cuboid target
        # Since a circle has the most area compared to its perimeter, it
        # will always erode last out of any shape with the same area
        # Therefor, this is a fairly good classifier for the two shapes.

        grey = self.greyscale(img)

        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(grey.copy(), kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=3)

        return eroded


    def draw_spot_at(self, img, pos):
        img[pos[1], pos[0]] = [255, 255, 255]
        img[pos[1]+1, pos[0]] = [0, 0, 0]
        img[pos[1]-1, pos[0]] = [0, 0, 0]
        img[pos[1], pos[0]+1] = [0, 0, 0]
        img[pos[1], pos[0]-1] = [0, 0, 0]

    # 128 pixels = 5m then 25.6 pixels to a metre.

    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        size = (400, 400)
        centre1 = (400, 500)
        centre2 = (400, 500)
        yellow_sphere_location = np.array([0, 0, 0.5])

        # Crop the images so we're not doing unnecessary work
        img1 = self.cv_image1[
               centre1[1] - size[1] / 2:centre1[1] + size[1] / 2,
               centre1[0] - size[0] / 2:centre1[0] + size[0] / 2
               ]
        img2 = self.cv_image2[
               centre2[1] - size[1] / 2:centre2[1] + size[1] / 2,
               centre2[0] - size[0] / 2:centre2[0] + size[0] / 2
               ]

        positions1 = self.get_positions(img1, 1)
        positions2 = self.get_positions(img2, 2)

        master_positions = np.zeros(shape=(5,3))

        for i, pos in enumerate(positions1):
            master_positions[i] = np.array([-1, pos[0], pos[1]])

            self.draw_spot_at(img1, pos)

        for i, pos in enumerate(positions2):
            old_pos = master_positions[i].copy()
            master_positions[i] = np.array([pos[0], old_pos[1], (old_pos[2] + pos[1])/2])

            self.draw_spot_at(img2, pos)

            if abs(old_pos[2] - pos[1]) > 20 and pos[1] != -1 and old_pos[2] != -1:
                print("Warning. ", old_pos[2], pos[1])

        world_center = master_positions[0].copy()
        for i in range(5):
            master_positions[i] -= world_center

        master_positions /= 25.6

        for i in range(5):
            master_positions[i][2] *= -1
            master_positions[i] += yellow_sphere_location

        cv2.imshow('window1', np.asarray(img1))
        cv2.imshow('window2', np.asarray(img2))
# Our Y data sometimes jumps to -8

        cv2.waitKey(1)
        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(img2, "bgr8"))

            self.target_x_pub.publish(master_positions[4][0])
            self.target_y_pub.publish(master_positions[4][1])
            self.target_z_pub.publish(master_positions[4][2])
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
