import numpy as np

ACTIONS_ALL = {
   0: 'LANE_LEFT',
   1: 'IDLE',
   2: 'LANE_RIGHT',
   3: 'FASTER',
   4: 'SLOWER'
   }

CHANGE_LANE = [0, 2]


class BaselineAgent:

   def __init__(self, env):
      self.env = env

   def BasePolicy(self, state, th = 20, epsilon = 0.2):

      #Positions of all the other vehicles
      x, y = state[1:, 1], state[1:, 2]

      #Identify vehicles in front of ego vehicle
      xBool = (x > 0)
      x, y = x[xBool], y[xBool]

      forewardDistance =x
      closestVehicle = np.argmin(forewardDistance)

      #print(f"Ego Row: {state[0]}") 
      #print(f"First Other Row: {state[1]}")

      #If no one is close
      if forewardDistance[closestVehicle] > th:
         return self.env.unwrapped.action_type.actions_indexes["FASTER"]
      
      else:
         #Look at the y position of the closest vehicle
         yClosest = y[closestVehicle]

         if np.abs(yClosest) < epsilon:
            availableActions = self.env.unwrapped.action_type.get_available_actions()

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