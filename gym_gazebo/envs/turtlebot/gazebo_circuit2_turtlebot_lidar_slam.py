import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboCircuit2TurtlebotLidarSlamEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(5) #F,L,R,NL, NR
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def discretize_observation(self,data,angle_range=np.pi/6):
        discretized_ranges = []
        hfov = abs(data.angle_max - data.angle_min)
        len_ranges = len(data.ranges)
        rays_in_angle_range = int(len_ranges * angle_range / hfov)
        central_start_pos = right_stop_pos = int((len_ranges - rays_in_angle_range) / 2)
        right_start_pos = max(central_start_pos - rays_in_angle_range, 0)
        central_stop_pos = left_start_pos = int((len_ranges + rays_in_angle_range) / 2)
        left_stop_pos = min(central_stop_pos + rays_in_angle_range, len_ranges)


        right2_start_pos = max(int((-np.pi/2 - data.angle_min) * len_ranges / hfov), 0)
        left2_stop_pos = len_ranges - right_start_pos
        #left_stop_pos = min(left_start_pos+rays_in_angle_range, central_start_pos)
        #right_start_pos = max(right_stop_pos-rays_in_angle_range, central_stop_pos)
        right2_stop_pos = right_start_pos
        left2_start_pos = left_stop_pos

        min_range = 0.2
        done = False
        center = min(data.ranges[central_start_pos:central_stop_pos])
        # left = sum(data.ranges[left_start_pos:left_stop_pos])/rays_in_angle_range
        # right = sum(data.ranges[right_start_pos:right_stop_pos])/rays_in_angle_range
        left = min(data.ranges[left_start_pos:left_stop_pos])
        right = min(data.ranges[right_start_pos:right_stop_pos])
        left2 = min(data.ranges[left2_start_pos:left2_stop_pos])
        right2 = min(data.ranges[right2_start_pos:right2_stop_pos])
        if center == float ('Inf') or np.isinf(center):
            discretized_ranges.append(100)
        else:
            discretized_ranges.append(int(center*5))
        if left == float ('Inf') or np.isinf(left):
            discretized_ranges.append(100)
        else:
            discretized_ranges.append(int(left*5))
        if right == float ('Inf') or np.isinf(right):
            discretized_ranges.append(100)
        else:
            discretized_ranges.append(int(right*5))
        if left2 == float ('Inf') or np.isinf(left2):
            discretized_ranges.append(100)
        else:
            discretized_ranges.append(int(left2*5))
        if right2 == float ('Inf') or np.isinf(right2):
            discretized_ranges.append(100)
        else:
            discretized_ranges.append(int(right2*5))

        if min(center, left, right) < min_range:
            done = True

        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 3: #ROTATE LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 4: #ROTATE RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.3
            self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data, 0.3)

        if not done:
            if action == 0:
                reward = 5
            elif action == 3 or action == 4:
                reward = 0
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.discretize_observation(data, 0.3)

        return state
