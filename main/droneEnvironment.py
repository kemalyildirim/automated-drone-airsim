import airsim
import time
import numpy as np
from PIL import Image

def vector3rToNpArray(vctr):
    if isinstance(vctr, airsim.Vector3r):
        res = np.array([vctr.x_val, vctr.y_val, vctr.z_val])
    else:
        res = np.array(vctr)
    return res

def npDistance(pos1, pos2):
    point1, point2 = vector3rToNpArray(pos1), vector3rToNpArray(pos2)
    dist = np.linalg.norm(point1 - point2)
    return dist

goal_threshold = 20
#goal_threshold = 25
np.set_printoptions(precision=3, suppress=True)

class DroneEnvironment:
    #aim = [32, 38, -4]
    def __init__(self, start = [0, 0, -5], aim = [76, 59, 10]):
        self.start = np.array(start)
        self.aim = np.array(aim)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.threshold = goal_threshold

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2], 5)

    def Land(self):
        '''
        cur_pos = self.client.simGetGroundTruthKinematics().position
        if not cur_pos.z_val == 0:
            self.client.moveToPositionAsync(cur_pos.x_val, cur_pos.y_val, 0, 2)
            time.sleep(1)
            self.client.moveByVelocityAsync(0, 0, 0, 1)
            return False
        else:
            return True
        '''
        self.client.landAsync().join()
        #landed = self.client.getMultirotorState().landed_state
        ##Can't rely on getMultirotorState().
        return landed
    
    def isDone(self):
        pos = self.client.simGetGroundTruthKinematics().position
        if npDistance(self.aim, pos) < self.threshold:
            return True
        return False

    def getImage(self):
        responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True, False)])                
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 * img1d
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((224, 224)).convert('L')) 
        return im_final


    def moveByDist(self, diff):
        self.client.moveByVelocityAsync(diff[0], diff[1], diff[2], 1 ,drivetrain = airsim.DrivetrainType.ForwardOnly)
        # TO DO:
        ##move with diff speed for 1 sec.
        time.sleep(1)
        self.client.moveByVelocityAsync(0,0,0,1,drivetrain = airsim.DrivetrainType.ForwardOnly)

    def render(self, extra1 = "", extra2 = ""):
        pos = vector3rToNpArray(self.client.simGetGroundTruthKinematics().position)
        goal = npDistance(self.aim, pos)
        print(extra1, "distance: ", int(goal), "position: ", pos.astype("int"), extra2)


class GridWorld(DroneEnvironment):
    #aim = [32,38,-4]
    def __init__(self, start = [0,0,-5], aim = [76, 59, 10], scaling_factor = 5):
        DroneEnvironment.__init__(self, start, aim)
        self.scaling_factor = scaling_factor

    def interpret_action(self, action):
        scaling_factor = self.scaling_factor
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            quad_offset = (-scaling_factor, 0, 0)
        elif action == 5:
            quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -scaling_factor)
        return np.array(quad_offset).astype("float64")

    def rewardf(self, state, state_):
        dis = npDistance(state[0:3], self.aim)
        dis_ = npDistance(state_[0:3], self.aim)
        reward = dis - dis_
        reward -= 1
        return reward

    def step(self, action):
        diff = self.interpret_action(action)
        DroneEnvironment.moveByDist(self, diff)

        pos_ = vector3rToNpArray(self.client.simGetGroundTruthKinematics().position)
        vel_ = vector3rToNpArray(self.client.simGetGroundTruthKinematics().linear_velocity)
        state_ = np.append(pos_, vel_)
        pos = self.state[0:3]

        info = None
        done = False
        reward = self.rewardf(self.state, state_)
        if action == 0:
            reward -= 10
        if self.isDone():
            done = True
            reward = 100
            info = "success"
        if self.client.simGetCollisionInfo().has_collided:
            reward = -100
            done = True
            info = "collision"
        #Distance threshold. checkpoint => 200
        if (npDistance(pos_, self.aim) > 110):
            reward = -100
            done = True
            info = "out of range"

        self.state = state_ # #STATE. 'state' is previous state, state_ is current state
        return state_, reward, done, info

    def reset(self):
        pos = vector3rToNpArray(self.client.simGetGroundTruthKinematics().position)
        vel = vector3rToNpArray(self.client.simGetGroundTruthKinematics().linear_velocity)
        state = np.append(pos, vel)
        self.state = state
        DroneEnvironment.reset(self)
        return state