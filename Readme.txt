To run the code you need just to run the .py file. There are different variables that will give you the possibility to train and test different approaches.
The variable UptadeMethod allows to choose the target network update method. It can be either a periodic update or Polyak update.
The variable Qlearning allows to choose the method : either DQN or DDQN.
The TARGET_UPDATE_Training_Steps determines for the periodic update the period for changing the target net. 
The variable TAU is the constant in the polyak equation.
StepsToTrain determines the number of frames after which we will train the network on a batch of simples from the memory.
