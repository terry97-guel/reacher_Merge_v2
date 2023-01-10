# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import random as rd 
from gym import utils
from gym.envs.mujoco.reacher import ReacherEnv

import sys
from pathlib import Path

from envs.PDController import PD_Controller

'''Reward Definition'''
def dist2quadrant(position, quadrant):
    theta = -np.radians(90) * (quadrant)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    position_ = np.matmul(R,position[:2])

    x,y = position_[0],position_[1]
    if x>=0 and y>=0:
        reward = 2
    elif x<=0 and y>=0:
        reward = -np.abs(x)*10
    elif x>=0 and y<=0:
        reward = -np.abs(y)*10
    elif x<=0 and y<=0:
        reward = -np.sqrt(x**2+y**2)*10
    else: raise Exception("Code Error: can not handle coner cases")
    
    return reward


class CustomReacherEnv(ReacherEnv, utils.EzPickle):
    def __init__(self,Kp=0.01*60,Kd=0.001*60,
                jointlimit=[3.14,2.22]):
        # initalize Mujoco
        utils.EzPickle.__init__(self)
        ReacherEnv.__init__(self)

        # Limit
        joint1_limit,joint2_limit      =    jointlimit
        self.minDegree                 =    np.array([-joint1_limit, -joint2_limit])
        self.maxDegree                 =    np.array([ joint1_limit,  joint2_limit])
        self.minTorque                 =    self.action_space.low[0]
        self.maxTorque                 =    self.action_space.high[1]

        # PD Controller
        self.PD_Controller             =    PD_Controller(Kp=Kp, Kd=Kd, dt=self.dt)
        
        

    def step_trajectory(self,goal_joint_trajectory,RENDER=False):
        joint_trajectory = []
        for goal_joint in goal_joint_trajectory:
            curr_joint = self.get_joint_value()
            diff_joint = curr_joint-goal_joint
             
            torque = self.PD_Controller.control(diff_joint)
            
            # handle nan case 
            if np.isnan(torque).any(): torque = 0; print("Warning: nan in torque...setting torque value to zero")
            # clip to limited range
            torque     = np.clip(torque, self.minTorque, self.maxTorque)
            torque     = torque.flatten()
            
            # run simulation step
            self.step(torque)
            if RENDER: self.render()
            
            # append
            joint_trajectory.append(self.get_joint_value())
        
        joint_trajectory        =    np.array(joint_trajectory)
        # Get results
        last_position           =    self.get_endeffector_position()
        last_quadrant           =    get_quadrant(last_position)
        reward                  =    dist2quadrant(last_position,self.target_quadrant)
        
        
        assert np.linalg.norm(joint_trajectory[-1] - goal_joint_trajectory[-1])<1e-1
        return reward,last_quadrant,last_position,np.array(joint_trajectory)
        
    def get_endeffector_position(self):
        return np.array(self.get_body_com("fingertip")[:2])
    
    
    def get_joint_value(self):
        joint_value_ = self.sim.data.qpos[:2]
        return np.array(joint_value_)

    """ Overide Mujoco function """
    def step(self, torque):
        # Simulation
        self.do_simulation(torque, self.frame_skip)
        ob = self._get_obs()
        done = False
        # reward_dist = self.reward_measure()
        reward_dist = 0
        reward = reward_dist
        return ob, reward, done, dict(reward_dist=reward_dist)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self,target_quadrant):
        # initalize arm
        qpos = self.init_qpos
        qvel = self.init_qvel
        
        # initalize target quadrant
        quadrant_visualize_ball_candidate =[
            np.array([0.1, 0.1]),
            np.array([-0.1, 0.1]),
            np.array([-0.1, -0.1]),
            np.array([0.1, -0.1])
            ]
        
        self.target_quadrant = target_quadrant
        quadrant_visualize_ball_pos = quadrant_visualize_ball_candidate[target_quadrant]
        qpos[-2:] = quadrant_visualize_ball_pos
        qvel[-2:] = 0
        
        # initalize PD_Controller
        self.PD_Controller.clear()
        
        # set_state
        self.set_state(qpos, qvel)
        
        return self._get_obs()
    
    def _get_obs(self):
        joint_value = self.sim.data.qpos.flat[:2]
        return dict(joint_value=joint_value, joint_value_vel = self.sim.data.qvel.flat[:2], quadrant_visualize_ball_pos=self.sim.data.qpos.flat[2:])
    
    def _get_state_vector(self):
        theta = self.sim.data.qpos[:2]
        return np.concatenate(
            [theta,
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")])

def get_quadrant(last_position):
    x,y = last_position[0],last_position[1]
    if x>=0 and y>=0: return 0
    elif x<=0 and y>=0: return 1
    elif x<=0 and y<=0: return 2
    elif x>=0 and y<=0: return 3
    else: raise Exception("Invalid input")

if __name__ == "__main__":
    env = CustomReacherEnv() 
    # Check state
    for i in range(1000):
        env.sim.data.qpos[:2] =[ 0.1865625,  -0.05842637]
        env.render()
    