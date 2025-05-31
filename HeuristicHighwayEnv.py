from highway_env.envs.highway_env import HighwayEnv
from random import randint
import numpy as np
import matplotlib.pyplot as plt

def agent(observations):
   #action = randint(0, 4)

   '''
   ACTION NUMBER
   0: 'LANE_LEFT',
   1: 'IDLE',
   2: 'LANE_RIGHT',
   3: 'FASTER',
   4: 'SLOWER'
   '''

   observations=np.array(observations)
   agent_speed=observations[0,2]

   #detect if a vehicle is in the same lane as the agent
   if round(np.min(abs(observations[1:,1])))!=0:
       same_lane_vehicle=False
   else:
       same_lane_vehicle=True

       #speed of same lane vehicle
       same_lane_vehicle_x_mat=np.argmin(np.abs(observations[1:,1]))+1 #dont forget +1
       same_lane_vehicle_speed=observations[same_lane_vehicle_x_mat,2]+agent_speed
       same_lane_vehicle_x_pos=observations[same_lane_vehicle_x_mat,0]

       print(f'Ahead Vehicle Speed: {same_lane_vehicle_speed}')
   print(f'Agent Speed: {agent_speed}')

   #determine if changing lanes is safe
   lane_change_left_safe=True
   lane_change_right_safe=True
   lane_change_buffer=(3.5*agent_speed)-55 #a function of agent speed y=3.5x-55
   if np.argwhere(np.round(observations[1:,1])==-4).size!=0: #if left adjacent vehicle exists
       adjacent_vehicle_left_x_mat=int(np.argwhere(np.round(observations[1:,1])==-4)[0])+1
       if abs(observations[adjacent_vehicle_left_x_mat,0])<lane_change_buffer: #if left vehicle is within buffer, left lane change is not safe
           lane_change_left_safe=False
   if np.argwhere(np.round(observations[1:,1])==4).size!=0: #if right adjacent vehicle exists
       adjacent_vehicle_right_x_mat=int(np.argwhere(np.round(observations[1:,1])==4)[0])+1
       if abs(observations[adjacent_vehicle_right_x_mat,0])<lane_change_buffer: #if right vehicle is within buffer, right lane change is not safe
           lane_change_right_safe=False
   print(f'Left Lane Safe: {lane_change_left_safe}')
   print(f'Right Lane Safe: {lane_change_right_safe}')

   #determine an empty lane
   closest_unoccupied_lane_relative=None
   occupied_lanes=np.round(observations[1:,1]+observations[0,1])/4 #gets the occupied lanes and lane numbers 1-4
   occupied_lanes=occupied_lanes.astype(int)
   all_lanes_index=set(range(0,4))
   unoccupied_lanes=np.array(list(set(occupied_lanes).symmetric_difference(all_lanes_index))) #gets the lane index of unoccupied lanes
   if unoccupied_lanes.size==0: #if there are no empty lanes
       empty_lanes=False
   else:
       empty_lanes=True
       unoccupied_lanes_relative=(unoccupied_lanes*4)-round(observations[0,1]) #distance of the empty lanes relative to the agent
       closest_unoccupied_lane_relative=min(unoccupied_lanes_relative,key=lambda x:abs(x))
   print(f'Closest Empty Lane: {closest_unoccupied_lane_relative}')

   #if car is ahead, agent is too close, agent is faster than car ahead, and agent is not at minimum speed, slow down
   forward_buffer_distance=(8*agent_speed)-135 #function of agent speed y=8x-135
   if same_lane_vehicle==True and same_lane_vehicle_x_pos<forward_buffer_distance and agent_speed>same_lane_vehicle_speed and round(agent_speed)!=20:
       action=4
       return action

   #if there is an empty lane, change lane towards empty lane
   if empty_lanes==True and closest_unoccupied_lane_relative!=0:
       if closest_unoccupied_lane_relative<0 and lane_change_left_safe==True:
           action=0
           return action
       if closest_unoccupied_lane_relative>0 and lane_change_right_safe==True:
           action=2
           return action

   #determine lane of furthest vehicle and switch to its lane
   furthest_vehicle_lane=round(observations[4,1])
   furthest_vehicle_lane_num=np.argwhere(np.round(observations[1:,1])==furthest_vehicle_lane) #gives x_mat of vehicle(s) in lane with the furthest vehicle
   if empty_lanes==False and closest_unoccupied_lane_relative==None: #if there are no empty lanes and closest_unoccupied_lane_relative is None
       if furthest_vehicle_lane_num.size==1: #1 vehicle in lane
           if furthest_vehicle_lane<0 and lane_change_left_safe==True:
               action=0
               return action
           if furthest_vehicle_lane>0 and lane_change_right_safe==True:
               action=2
               return action
       if furthest_vehicle_lane_num.size==2: #2 vehicles in lane
           if observations[furthest_vehicle_lane_num[0]+1,0]<0: #if second vehicle is behind agent
               if furthest_vehicle_lane<0 and lane_change_left_safe==True:
                   action=0
                   return action
               if furthest_vehicle_lane>0 and lane_change_right_safe==True:
                   action=2
                   return action

   #if there are no cars ahead, speed up
   if same_lane_vehicle==False:
       action=3
       return action
   ''''
   #if car is ahead and agent speed is > and is within forward_buffer_distance, slow down
   if same_lane_vehicle==True and agent_speed>same_lane_vehicle_speed and same_lane_vehicle_x_pos<forward_buffer_distance:
       action=4
       return action
   '''
   #if car is ahead, agent speed is <, and agent is outside buffer zone, speed up
   if same_lane_vehicle==True and agent_speed<same_lane_vehicle_speed and same_lane_vehicle_x_pos>forward_buffer_distance:
       action=3
       return action

   #if ahead car is under agent minimum speed and agent is within buffer zone, change to safe lane
   if same_lane_vehicle==True and same_lane_vehicle_speed<20 and same_lane_vehicle_x_pos<forward_buffer_distance:
       rand_num=randint(2,3) #random number for changing to random lane.
       if lane_change_left_safe==True and rand_num%2==0:
           action=0
           return action
       if lane_change_right_safe==True and rand_num%2==1:
           action=2
           return action

   action=1
   return action


if __name__ == "__main__":
   config = {
       "lanes_count": 4,
       "vehicles_count": 50,
       "duration": 100,  # [s]
       "initial_spacing": 2,
       "simulation_frequency": 15,  # [Hz]
       "policy_frequency": 0.25,  # [Hz]
       "render_agent": True,
       #"show_trajectories": True,
       #"manual_control": True,
       "observation":{
           "type": "Kinematics",
           "features": ["x", "y", "vx"],
           "normalize": False,
           "absolute": False,
       },
   }

   env = HighwayEnv(render_mode="rgb_array")
   #env.vehicle.DEFAULT_INITIAL_SPEEDS=[20,20]
   env.configure(config)

   for _ in range(5):
       obs, info = env.reset()
       done = truncated = False
       while not (done or truncated):
           action = agent(observations=obs)
           print(f'Action: {action}')
           obs, reward, done, truncated, info = env.step(action)
           print(obs)
           env.render()

           #plt.imshow(env.render())
           #plt.show()