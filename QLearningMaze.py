import numpy as np
import pandas as pd
import time
import random

np.random.seed(2)  # reproducible

N_STATES = (10,10)   # the length of the 1 dimensional world
ACTIONS = ['left','right','up','down']     # available actions
EPSILON = 0.7   # greedy police
ALPHA = 0.01     # learning rate
GAMMA = 0.99    # discount factor
MAX_EPISODES = 100   # maximum episodes
FRESH_TIME = 0.00    # fresh time for one move
action_sym={'left':'◀','right':'▶','up':'▲','down':'▼'}

env_list=[['-' for rows in range(N_STATES[0])] for col in range(N_STATES[0])]
#create random obstacles
obst_num=30
obst_cords=[[random.randint(1,9) for i in range(2)] for j in range(obst_num)] #cannot be 0,0 because agent starts there
for i in range(obst_num):
   env_list[obst_cords[i][0]][obst_cords[i][1]]='□'

#create random terminal
term_cords=(random.randint(0,N_STATES[0]-1),random.randint(0,N_STATES[1]-1))
env_list[term_cords[0]][term_cords[1]]='T'


def build_q_table(n_states, actions):
   table = pd.DataFrame(
       np.zeros((n_states, len(actions))),     # q_table initial values
       columns=actions,    # actions's name
   )
   # print(table)    # show table
   return table


def choose_action(state, q_table):
   # This is how to choose an action
   state_actions = q_table.iloc[state, :]
   if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
       action_name = np.random.choice(ACTIONS)
   else:   # act greedy
       action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
   return action_name


def get_env_feedback(S, A):
   # This is how agent will interact with the environment
   global term_cords

   Sx,Sy=S

   if A=='right': # move right
       Sy_=Sy+1
       Sx_=Sx
   if A=='left':
       Sy_=Sy-1
       Sx_=Sx

   if A=='up':
       Sx_=Sx-1
       Sy_=Sy
   if A=='down':
       Sx_=Sx+1
       Sy_=Sy

   Sx_=max(0,min(Sx_,N_STATES[0]-1))
   Sy_=max(0,min(Sy_,N_STATES[1]-1))

   for i in range(obst_num):
       if Sx_==obst_cords[i][0] and Sy_==obst_cords[i][1]:
           Sx_=Sx
           Sy_=Sy

   S_=(Sx_,Sy_)

   if S_==term_cords:
       S_='terminal'
       R=1
   else:
       R=0

   return S_, R

def update_env(S, episode, step_counter):
   # This is how environment be updated
   global env_list

   if S == 'terminal':
       interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
       print('\n{}'.format(interaction), end='')
       #time.sleep(2)
       #print('\n                                ', end='')
   else:
       #env_list[S[0]][S[1]] = 'o'
       interaction = '\n'.join([''.join(row) for row in env_list])
       #print('\n{}'.format(interaction), end='')
       time.sleep(FRESH_TIME)


def rl():
   # main part of RL loop
   q_table = build_q_table(N_STATES[0]*N_STATES[1], ACTIONS)
   for episode in range(MAX_EPISODES):
       global EPSILON

       step_counter = 0
       S = (0,0)
       is_terminated = False
       update_env(S, episode, step_counter)
       while not is_terminated:
           q_table_Spos=(S[0]*10)+S[1]

           A = choose_action(q_table_Spos, q_table)
           S_, R = get_env_feedback(S, A)  # take action & get next state and reward

           q_table_S_pos=(S_[0]*10)+S_[1]

           q_predict = q_table.loc[q_table_Spos, A]
           if S_ != 'terminal':
               q_target = R + GAMMA * q_table.iloc[q_table_S_pos, :].max()   # next state is not terminal
           else:
               q_target = R     # next state is terminal
               is_terminated = True    # terminate this episode

           #
           if episode==MAX_EPISODES-1:
               env_list[S[0]][S[1]]=action_sym[A]

           q_table.loc[q_table_Spos, A] += ALPHA * (q_target - q_predict)  # update
           S = S_  # move to next state

           update_env(S, episode, step_counter+1)
           step_counter += 1
       EPSILON*=1.005
   return q_table


if __name__ == "__main__":
   q_table = rl()
   print('\r\nQ-table:\n')
   print(q_table)
   interaction = '\n'.join(['  '.join(row) for row in env_list])
   print(interaction)