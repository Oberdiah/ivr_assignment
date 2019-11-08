#!/usr/bin/env python
# rostopic pub -1 /robot/joint2_position_controller/command std_msgs/Float64 "data: 1.0"
import math
import time

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
        self.last_known_locations = np.zeros(shape=(5,3))

        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)

        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1, queue_size=1, buff_size=2**24)

        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2, queue_size=1, buff_size=2**24)

        self.target_x_pub = rospy.Publisher("target_x", Float64, queue_size=1)
        self.target_y_pub = rospy.Publisher("target_y", Float64, queue_size=1)
        self.target_z_pub = rospy.Publisher("target_z", Float64, queue_size=1)

        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=1)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=1)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=1)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=1)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

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

        if img_num == 1:
            cv2.imshow('orange target', targets[4])

        positions = [0, 0, 0, 0, 0]

        for i in range(5):
            M = cv2.moments(targets[i])
            if M['m00'] == 0:
                positions[i] = (-1,-1)
                # print("Warning. Sphere " + str(i) + " has been lost in image " + str(img_num))
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

        # Removing all thin (number of lit pixels in a row < 10) sections of the image
        # is a very effective way of removing all traces of the rectangle while still maintaining as much
        # of the circle as possible
        cython_functions.remove_thin_bits(img, 10, 2)
        grey = self.greyscale(img)
        # kernel = np.ones((3, 3), np.uint8)
        # dilated = cv2.dilate(grey.copy(), kernel, iterations=1)
        # eroded = cv2.erode(grey, kernel, iterations=1)

        return grey


    def draw_spot_at(self, img, pos):
        try:
            img[pos[1], pos[0]] = [255, 255, 255]
            img[pos[1]+1, pos[0]] = [0, 0, 0]
            img[pos[1]-1, pos[0]] = [0, 0, 0]
            img[pos[1], pos[0]+1] = [0, 0, 0]
            img[pos[1], pos[0]-1] = [0, 0, 0]
        except:
            return

    # 128 pixels = 5m then 25.6 pixels to a metre.

    # Recieve data and save it for camera 1's callback.
    def callback1(self, data):
        self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8").copy()

    # Recieve data from camera 1, process it, and publish
    def callback2(self, data):
        self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8").copy()

        size = (400, 400)
        centre1 = (400, 500)
        centre2 = (400, 500)
        yellow_sphere_location = np.array([0, 0, 0.5])

        # Crop the images so we're not doing unnecessary work
        img1 = self.cv_image1[
               centre1[1] - size[1] / 2:centre1[1] + size[1] / 2,
               centre1[0] - size[0] / 2:centre1[0] + size[0] / 2
               ].copy()
        img2 = self.cv_image2[
               centre2[1] - size[1] / 2:centre2[1] + size[1] / 2,
               centre2[0] - size[0] / 2:centre2[0] + size[0] / 2
               ].copy()

        positions1 = self.get_positions(img1, 1)
        positions2 = self.get_positions(img2, 2)

        master_positions = np.zeros(shape=(5,3))

        for i, pos in enumerate(positions1):
            master_positions[i] = np.array([-1, pos[0], pos[1]])

            self.draw_spot_at(img1, pos)

        for i, pos in enumerate(positions2):
            old_pos = master_positions[i].copy()

            avg = (old_pos[2] + pos[1])/2 if old_pos[2] != -1 and pos[1] != -1 else -1

            master_positions[i] = np.array([pos[0], old_pos[1], avg])

            self.draw_spot_at(img2, pos)

            # if abs(old_pos[2] - pos[1]) > 20 and pos[1] != -1 and old_pos[2] != -1:
            #     print("Warning. ", old_pos[2], pos[1])

        for i in range(5):
            for j in range(3):
                if master_positions[i][j] == -1:
                    master_positions[i][j] = self.last_known_locations[i][j]
                    # print("We didn't have position data for sphere " + str(i) + ", so assumed last known position.")
                else:
                    self.last_known_locations[i][j] = master_positions[i][j]

        world_center = master_positions[0].copy()
        for i in range(5):
            master_positions[i] -= world_center

        # Convert pixel distances into real world distances
        master_positions /= 25

        for i in range(5):
            # Since the y coordinate of the images is flipped, we need to flip it again to
            # get back to sensible real world results.
            master_positions[i][2] *= -1
            master_positions[i] += yellow_sphere_location

            # This helps the calculations be more accurate, but can't be justified so it's unused
            # master_positions[i][1] *= 4.0/5

        # target_pos = master_positions[4]-master_positions[1]
        # joint2 = -math.atan(target_pos[1]/target_pos[2])
        # target_pos_magnitude = math.sqrt(target_pos[2]*target_pos[2] + target_pos[1]*target_pos[1])
        # transformed_target_pos = [target_pos[0], target_pos_magnitude * math.copysign(1, target_pos[1]), 0]
        # joint3 = math.atan(transformed_target_pos[0]/transformed_target_pos[1])
        # joint1 = 0
        # joint4 = 0

        actual_target_pos = [4, 4, 4]
        test_angle = time.time()/2 % (math.pi * 2) - math.pi
        x = actual_target_pos[0]
        y = actual_target_pos[1]
        joint1 = -test_angle
        test_angle = test_angle if test_angle < 0 else math.pi-test_angle
        c = math.cos(test_angle+math.pi*3/4)
        s = math.sin(test_angle+math.pi*3/4)
        target_pos = [x * c - y * s, x * s + y * c, actual_target_pos[2]]
        joint2 = -math.atan(target_pos[1] / target_pos[2])
        target_pos_magnitude = math.sqrt(target_pos[2] * target_pos[2] + target_pos[1] * target_pos[1])
        transformed_target_pos = [target_pos[0], target_pos_magnitude * math.copysign(1, target_pos[1]), 0]
        joint3 = math.atan(transformed_target_pos[0] / transformed_target_pos[1])
        joint4 = math.pi/2


        cv2.imshow('window1', np.asarray(img1))
        cv2.imshow('window2', np.asarray(img2))

        cv2.waitKey(1)

        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(img2, "bgr8"))

            self.target_x_pub.publish(master_positions[4][0])
            self.target_y_pub.publish(master_positions[4][1])
            self.target_z_pub.publish(master_positions[4][2])

            self.robot_joint1_pub.publish(joint1)
            self.robot_joint2_pub.publish(joint2)
            self.robot_joint3_pub.publish(joint3)
            self.robot_joint4_pub.publish(joint4)
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
