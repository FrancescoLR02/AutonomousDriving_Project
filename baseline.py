import numpy as np

ACTIONS_ALL = {
   0: 'LANE_LEFT',
   1: 'IDLE',
   2: 'LANE_RIGHT',
   3: 'FASTER',
   4: 'SLOWER'
   }

CHANGE_LANE = [0, 2]
SLOWER = [4]


class BaselineAgent:

   def __init__(self, env):
      self.env = env

   def BasePolicy(self, state, th = 0.1, epsilon = 0.2):

      #Positions of all the other vehicles
      x, y = state[1:, 1], state[1:, 2]

      #Identify vehicles in front of ego vehicle
      xBool = (x > 0)
      forewardDistance, y = x[xBool], y[xBool]

      closestVehicle = np.argmin(forewardDistance)

      #If no one is close
      if forewardDistance[closestVehicle] > th:
         return self.env.unwrapped.action_type.actions_indexes["FASTER"]
      
      else:
         #Look at the y position of the closest vehicle
         yClosest = y[closestVehicle]

         if np.abs(yClosest) < epsilon:
            availableActions = self.env.unwrapped.action_type.get_available_actions()

            if np.isin(SLOWER, availableActions):
               return self.env.unwrapped.action_type.actions_indexes["SLOWER"]

            #Check what action is in available actions
            changeLane = np.isin(CHANGE_LANE, availableActions)

            #If i can move either right or left
            if all(changeLane) == True:
               if(np.random.random()) > 0.5: 
                  return self.env.unwrapped.action_type.actions_indexes["LANE_LEFT"]
               else: return self.env.unwrapped.action_type.actions_indexes["LANE_RIGHT"]
            
            #If i can only move right or left
            else:
               if changeLane[0] == True:
                  return self.env.unwrapped.action_type.actions_indexes["LANE_LEFT"]
               else: return self.env.unwrapped.action_type.actions_indexes["LANE_RIGHT"]


   def RandomPolicy(self):

      return self.env.action_space.sample()