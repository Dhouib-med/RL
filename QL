from pathlib import Path
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import base64, io
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob
from tqdm import tqdm
import time
tqdm.pandas()
from gym.wrappers import Monitor
from datetime import timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def getlist(self):
        return list(self.memory)
    
    
class DQN(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.back = nn.Linear(inputs,256 )
        self.meduim=nn.Linear(256,256 )
        self.head = nn.Linear(256,outputs)

    def forward(self, x):
        x = x.to(device)
        x =F.relu(self.back (x))
        x =F.relu(self.meduim (x))
        x =self.head (x)
        return x
def GetEpsilon():
    return EPS_END + (EPS_START - EPS_END) *  math.exp(-1. * StepsDone/ EPS_DECAY)

def select_action(state):
    global GamesPlayed
    sample = random.random()
    eps_threshold = GetEpsilon()
    if sample > eps_threshold:
        with torch.no_grad():
            return target_net(state).max(1)[1].view(1, 1).detach()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    global doiprint
    if len(memory) < BATCH_SIZE:
        return
    else :
     if doiprint: 
        print("start")
        doiprint=False
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

 
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_net(state_batch).gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if Qlearning=="DQN" :
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    elif Qlearning=="DDQN":
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1,(policy_net(non_final_next_states).max(1)[1].unsqueeze(1))).squeeze(1).detach()

    #else :
        #MinValueOfBothQNets=torch.minimum(target_net(non_final_next_states).max(1)[0].detach(),policy_net(non_final_next_states).max(1)[0].detach())
        #next_state_values[non_final_mask] =MinValueOfBothQNets

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values.squeeze(1) , expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def Test(network,path,Name,PrintEpsilon=True,Iter=20):
        global scores
        global NbreOfTraining
        global Resolved
        rewards=0
        reward_final=0
        for i in range(Iter):
            observation=env.reset()
            observation=torch.tensor(observation).float().unsqueeze(0)
            for j in count():
                with torch.no_grad():
                    action=network(observation).max(1)[1].view(1, 1).detach()
                observation, reward, done, _ = env.step(action.item())
                observation=torch.tensor(observation).float().unsqueeze(0)
                rewards+=reward
                if done:
                    reward_final+=reward
                    break
                if not timelimit and j>Test_Time-2:
                    break
        scores.append(rewards/Iter)
        NbreOfTraining.append(int(StepsDone/StepsToTrain))
        print("\n For  {} model and Traning episode number {} Episode we got an average reward of {} and final reward of {}".format(Name,i_episode,rewards/Iter,reward_final/Iter ))
        if PrintEpsilon:
            print("\n epsilon is {}".format(GetEpsilon()))
        torch.save(network.state_dict(), path)
        if rewards/Iter>ResolvedScore:
            Resolved=True


def PlayAGame(optimize):
    global StepsDone
    observation=env.reset()
    observation=torch.tensor(observation).float().unsqueeze(0)
    for t in count():
        action = select_action(observation)
        observation_next, reward, done, _ = env.step(action.item())
        observation_next=torch.tensor(observation_next).float().unsqueeze(0)
        reward = torch.tensor([reward], device=device).float()
        if done:
           observation_next=None
        if not t==Test_Time:
            memory.push(observation, action, observation_next, reward)
        observation=observation_next
        if optimize :
            StepsDone=StepsDone+1
            if StepsDone%StepsToTrain==0 :
                optimize_model()
                if UptadeMethod=="Periodic":
                    for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
                elif UptadeMethod=="LinearSlowUptade" and StepsDone%(TARGET_UPDATE_Training_Steps*StepsToTrain)==0:
                    target_net.load_state_dict(policy_net.state_dict())
                    target_net.eval()

        if done:
            break
        if t>=TimeNeededForARandomSessionToEnd and not(timelimit) :
            break

def PlayASuccesfulGamewithouttraining():
    observation = env.reset()
    observation = torch.tensor(observation).float().unsqueeze(0)
    done=False
    if GameName=='ALE/Tennis-ram-v5':
        LastGame = ReplayMemory(5000)
        reward=-1
        while  (reward!=1):
            action = select_action(observation)
            observation_next, reward, done, _ = env.step(action.item())
            if reward ==-1:
                LastGame=ReplayMemory(5000)
                observation = env.reset()
                observation = torch.tensor(observation).float().unsqueeze(0)
            observation_next=torch.tensor(observation_next).float().unsqueeze(0)
            reward_torch = torch.tensor([reward], device=device).float()
            if done:
               observation_next=None
            LastGame.push(observation, action, observation_next, reward_torch)
            observation=observation_next
            if done:
                observation = env.reset()
                observation = torch.tensor(observation).float().unsqueeze(0)

    else:
        Time=2*Test_Time
        LastGame = ReplayMemory(Time)
        while  (not done):
            action = select_action(observation)
            observation_next, reward, done, _ = env.step(action.item())
            observation_next=torch.tensor(observation_next).float().unsqueeze(0)
            reward = torch.tensor([reward], device=device).float()
            if done:
               observation_next=None
            LastGame.push(observation, action, observation_next, reward)
            observation=observation_next
            if done:
                break
    for element in LastGame.getlist():
        memory.push(*element)

def OnlyTrain():
    global StepsDone
    Size=memory.__len__()
    print(Size)
    TrainTime=Size//10
    StepsDone = StepsDone + Size
    for i in range (TrainTime):
        optimize_model()
        if UptadeMethod == "Periodic":
            for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        elif UptadeMethod == "LinearSlowUptade" and StepsDone % (TARGET_UPDATE_Training_Steps * StepsToTrain) == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

def plot(NameOfModel,scores, NbreOfTraining,axisx):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot( NbreOfTraining, scores)
    plt.ylabel('Score for '+NameOfModel)
    plt.xlabel('Training {} '.format(axisx))
    plt.axhline(y=ResolvedScore, color='gray', linestyle='--')
    plt.show()
    import pickle
    with open(os.path.join(r"C:\Users\moham\Desktop\RL", NameOfModel,"scores.pickle"), "wb") as fp:  # Pickling
        pickle.dump(scores, fp)
    with open(os.path.join(r"C:\Users\moham\Desktop\RL", NameOfModel,"X.pickle"), "wb") as fp:  # Pickling
        pickle.dump(NbreOfTraining, fp)



def show_video_of_model(agent,NameOfGame,NameOfModel ):
    env = Monitor(gym.make(NameOfGame), os.path.join(r"C:\Users\moham\Desktop\RL", NameOfModel), force=True)
    observation = env.reset()
    done = False
    observation = torch.tensor(observation).float().unsqueeze(0)
    while not done:
        action = agent(observation).max(1)[1].view(1, 1).detach().item()
        observation, reward, done, _ = env.step(action)
        observation = torch.tensor(observation).float().unsqueeze(0)
    env.close()

for UptadeMethod in ["Periodic","LinearSlowUptade"]:
    for Qlearning in ["DQN","DDQN"]:
        EstimatedGameLength=200
        TimeNeededForARandomSessionToEnd=20000
        Test_Time=999 #normally the time limit if timelimit=True
        GameName='LunarLander-v2'
        ResolvedScore = 200
        timelimit=True
        if timelimit:
            env = gym.make(GameName)
        else:
            env = gym.make(GameName).env
        env.reset()
        BATCH_SIZE =128
        GAMMA = 0.99
        EPS_START = 1
        EPS_END = 0.01
        EPS_DECAY = int(EstimatedGameLength*6000)
        StepsToTrain = 10
        TARGET_UPDATE_Training_Steps = EstimatedGameLength/StepsToTrain*5 #Uptade each 5 game in average
        LR = 5e-4
        TAU = 5e-2
        Clipped2Q=False

        NameOfModel=UptadeMethod+"_"+Qlearning

        os.makedirs(os.path.join(r"C:\Users\moham\Desktop\RL",NameOfModel),exist_ok=True)
        path=os.path.join(r"C:\Users\moham\Desktop\RL",NameOfModel,"model_policy.pickle")
        Path_Target=os.path.join(r"C:\Users\moham\Desktop\RL",NameOfModel,"model_target.pickle")

        n_actions = env.action_space.n
        policy_net = DQN( env.observation_space.shape[0], n_actions).to(device)
        if os.path.exists(path):
            policy_net.load_state_dict(torch.load(path))
        policy_net.train()
        target_net = DQN( env.observation_space.shape[0], n_actions).to(device)
        if os.path.exists(Path_Target):
            target_net.load_state_dict(torch.load(Path_Target))
        target_net.eval()

        optimizer =  optim.Adam( policy_net.parameters(), lr=LR)

        Train=True
        num_episodes = 10505
        doiprint=True
        Times=[]
        memory = ReplayMemory(int(EstimatedGameLength*12000/4))
        FirstRandomGames=100
        PlayBeforeTraining=False
        ResolvedAlready = False
        TestEvery=500
        Solvedtime=10e12
        if Train:
            with tqdm(total=num_episodes, position=0, leave=True) as pbar:
                start=time.time()
                scores=[]
                NbreOfTraining = []
                StepsDone = 0
                i_episode=0
                Resolved=False
                for i in range(FirstRandomGames):
                    if PlayBeforeTraining:
                        PlayASuccesfulGamewithouttraining()
                        pbar.update()

                if PlayBeforeTraining:
                    OnlyTrain()
                    Episodes=range(FirstRandomGames,num_episodes)
                    print(GetEpsilon())
                else :
                    Episodes=range(num_episodes)
                for i_episode in Episodes:
                  pbar.update()
                  if GetEpsilon()<0.135 :
                      break
                  if i_episode%TestEvery==1:
                      iter=10
                      if i_episode>8400:
                          iter=200
                      TestTimeStart=time.time()
                      Test(policy_net,path,"policy",Iter=iter)
                      start=start+(time.time()-TestTimeStart)
                      Times.append(time.time() - start)
                      torch.save(target_net.state_dict(), Path_Target)


                  PlayAGame(True)
                  if Resolved and not ResolvedAlready :
                      Solvedtime=time.time()-start
                      print("Resolved in {}".format(timedelta(seconds=time.time()-start)))
                      ResolvedAlready=True


            plot(NameOfModel,scores,NbreOfTraining,"epoch")
            #plot(NameOfModel, scores,Times,"Time")
            print('Complete in {} and solved first time in {}'.format(timedelta(seconds=time.time()-start),timedelta(seconds=Solvedtime)))
            show_video_of_model(policy_net,GameName,NameOfModel)
