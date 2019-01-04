# OpenAI Gym's Lunar Lander Implementation

## Requirements

In order to run the code, you need to install the following:

* Python 3
* [NumPy](http://www.numpy.org/) 
* [TensorFlow](https://www.tensorflow.org/) 
* [OpenAI Gym](https://gym.openai.com/)

Specific instructions for each one of the above requirements can be found in their respective websites so I'll spare you any unnecessary details.

To run the code, execute the following command:

```
$ python3 lunar-lander.py
```

You can run the code in training or testing mode. To train the agent, make sure the `TRAINING` constant is set to `True`. Setting the `TRAINING` constant to `False` will run the simulator using the previously saved weights. I included as part of this repository a set of weights that can be used to test the agent without needing to train it first.

The output will display the episode number, the reward obtained on the current episode, the accumulated reward over the last 100 episodes, and the current value of epsilon:

```
Alpha: 0.0001 Gamma: 0.990 Epsilon 0.99941
0, -100.90, -100.90, 1.00
1, -148.32, -124.61, 1.00
2, -184.54, -144.59, 1.00
3, -188.26, -155.50, 1.00
4, -339.68, -192.34, 1.00
5, -114.51, -179.37, 1.00
6, -335.28, -201.64, 1.00
7, -166.12, -197.20, 1.00
8, -152.09, -192.19, 0.99
9, -400.20, -212.99, 0.99
10, -278.13, -218.91, 0.99
...
```

## Hyperparameters

The specific values for the hyperparameters that I explored during the implementation of the simulator are set up in the source code. Here are the more important ones:

```
LEARNING_RATE = [0.01, 0.001, 0.0001]
DISCOUNT_FACTOR = [0.9, 0.99, 0.999]
EPSILON_DECAY = [0.99910, 0.99941, 0.99954, 0.99973, 0.99987]
```

You can determine which values for these hyperparameters to use for your experiment while creating the agent (yes, you'd have to modify the code):

```
agent = Agent(training, LEARNING_RATE[2], DISCOUNT_FACTOR[1], EPSILON_DECAY[1])
```

## Introduction
Here is an implementation of a reinforcement learning agent that solves the OpenAI Gym’s Lunar Lander environment. This environment consists of a lander that, by learning how to control 4 different actions, has to land safely on a landing pad with both legs touching the ground. A successfully trained agent should be able to achieve a score equal to or above 200 on average over 100 consecutive runs.

I'm using an implementation of a Deep Q-Network (DQN) to solve the Lunar Lander environment. This implementation is inspired in the DQN described in (Mnih et al., 2015) where it was used to solve classic Atari 2600 games. Below I'm exploring the different decisions to construct a successful implementation, how some of the hyperparameters were selected, and the overall results obtained by the trained agent.

## Implementing the agent
The Lunar Lander environment has an infinite state space across 8 continuous dimensions, which makes the application of standard Q-learning not possible unless the space is discretized —which is inefficient and not practical for this problem. Instead, a DQN uses a Deep Neural Network (DNN) for approximating the `Q*(s, a)` function getting around the limitation of the standard Q-learning algorithm for infinite state spaces.

I used TensorFlow for the implementation of the DNN. The experience gained by the agent while acting in the environment was saved in a memory buffer, and a small batch of observations from this list was randomly selected and then used as the input to train the weights of the DNN —a process called "Experience Replay." The network uses an Adam optimizer, and its structure consists on an input layer with a node for each one of the 8 dimensions of the state space, a hidden layer with 32 neurons, and an output layer mapping each one of the 4 possible actions for the lander to take. The activation function between the layers is a rectifier (ReLU), and there's no activation function for the output layer. 

This setup, albeit very simple, didn't allow the agent to learn how to land consistently. In most of the experiments the agent wasn't able to surpass very low rewards, while in others the agent was able to obtain very high rewards for a bit, but quickly diverged out of control back to low rewards. As mentioned in (Lillicrap et al., 2015), directly implementing Q-learning with neural networks is unstable in many environments when the same network that's continuously updated is also used to calculate the target values. To get around this problem, the initial setup was modified and two networks were implemented; one network —referred as the Q network in the source code of the implementation— is continuously updated, while a second network —designated as the Q-target network— was used to predict the target value and only updated from the weights of the Q-target network after every episode. This change stabilized the performance of the agent, getting rid of the continuous divergence on most scenarios.

Finally, on every step, the agent follows an ε-greedy approach to select a random action with probability ε and the best action learned by the network with probability 1 - ε. The value of ε is continuously decayed after every episode to allow the agent first to spend most of its time exploring all the possibilities, while later switching to more exploitation of optimal policies.

## Hyperparameter search
The implemented solution relies on three particular hyperparameters that had to be carefully selected to achieve successful results: the learning rate (α) used by the DNN implementation, the discount factor (γ) of future rewards, and the decay rate of ε (ε-decay) to establish proper exploitation versus exploration balance.

To determine the best values for α and γ a grid search over a combination of three possible values for each was performed while keeping ε-decay constant. A total of 5,000 iterations were used to train the agent, so ε-decay was set to 0.99941 during the tunning of hyperparameters to decay the exploration rate to around 0.05% by the end of training with a 99.95% of exploitation rate (0.99941^5000 ≈ 0.05). This initial ε-decay seemed to be reasonable as a starting point to focus on the selection of α and γ, and after trying different values, there was no reason to select a different initial value because results weren't better.

The three values explored for the learning rate α were 0.01, 0.001, and 0.0001. A more exhaustive search through possible values would have undoubtedly led to a finer tuned learning rate, but the time limitation didn't allow it. These three values should allow for proper exploration of how the learning rate affects the convergence of the agent during training.

![Hyperparameter search — Learning rate and discount factor — Keeping ε-decay constant at 0.99941](https://github.com/svpino/lunar-lander/blob/master/images/chart1.png)

The three values explored for the discount factor γ were 0.9, 0.99, and 0.999. The discount factor indicates the ability of the agent to credit success to actions that will happen far in the future. A small value like γ = 0.9 will prevent the agent from properly crediting success after 10 steps in the future (1/(1-γ) = 1/(1-0.9) = 10) while a large value like γ = 0.999 will allow the agent to look all the way to 1,000 actions in the future (1/(1-γ) = 1/(1-0.999) = 1000). Since every episode of the implemented agent has a limit of 1,000 steps before being forcibly terminated, the grid search includes 0.9 to evaluate a myopic agent, 0.999 to evaluate a far-sighted agent, and 0.99 as a moderate value in between.

The figure above shows the accumulated reward over the last 100 episodes of the agent while trying the nine possible combinations of learning rate and discount factors over 5,000 episodes. (The best combination, α = 0.0001 and γ = 0.99, is highlighted in blue color from the rest.)

Notice how the results using α = 0.01 are very poor. A learning rate that's too large causes the network to constantly overshoot the minimum and prevents it from converging, and sometimes even causes divergences. Using a smaller learning rate α = 0.001 shows better results (especially when using the appropriate discount factor), but it's only with a smaller α = 0.0001 that convergence is reached. This value, when combined with the right discount factor, led to successful completion of the environment surpassing the 200 accumulated reward mark at around 3,600 iterations.

Looking at the discount factor, a value of 0.9 didn't lead to successful results because the agent was incapable of realizing rewards achieved when landing correctly —the agent couldn't solve the temporal credit assignment problem for more than 10 steps ahead—, so it learned how to fly, but never attempted to go beyond that. A much larger discount factor of 0.999, while allowing the agent to properly credit rewards obtained by landing correctly, didn't push the agent to land fast enough before the termination of an episode. The agent was able to fly indefinitely until the end of the episode with no rush to land. Finally, a moderate discount factor of 0.99 pushed the agent to land as fast as possible to collect the reward credited when landing —anything over 100 steps in the future wasn't credited. The best results from the experiments were using this value of discount factor.

With the best combination of learning rate and discount factor selected, an exploration of different values of ε-decay was performed. Besides the default value selected of 0.99941 —which offered 99.95% exploitation and 0.05% exploration by the end of training—, four more values were tested: 0.99910 (99.99% - 0.01%), 0.99954 (99.90% - 0.10%), 0.99973 (99.75% - 0.25%), and 0.99987 (99.50% - 0.50%). Here are the results:

![Hyperparameter search — Epsilon decay — Keeping the learning rate and discount factor constant at α=.0001 and γ=.99](https://github.com/svpino/lunar-lander/blob/master/images/chart2.png)

Notice that for larger values of ε-decay, the agent doesn't use what has learned frequently enough, so it's always exploring and doesn't take advantage of what it learns. For ε-decay = 0.99910 the agent quickly resorts to exploiting what it's learned but gets stuck because it can't explore other ways to increase its reward. The best result was using the initial value of 0.99941, which offers the right balance of exploration versus exploitation for the agent to uncover the entire state space.

There are other parameters necessary to make the lander work successfully. Due to time constraints, these values were selected by a trial and error method, and there wasn't an exhaustive exploration to tune them properly. These parameters were the following:

* The size of the replay buffer in memory was set at 250,000 to ensure that very early experiences from the agent would be discarded and only the experience collected from the latest 250 - 500 episodes would be used. 
* The size of the replay minibatch used to train the DNN was set at 32 although 64 showed similar results. Values didn't converge consistently using a minibatch of 16 or 128.
* The number of learning episodes was set at 5,000 to allow the agent to properly converge.
* The weights of the Q network were updated after every time step, albeit sacrificing performance.

## Agent results
Here is a look at the individual reward obtained by our agent while training using the best set of hyperparameters selected from the previous experiments. The accumulated reward is also displayed for context:

![Rewards per training episode — Using α=.0001, γ=.99, and ε = 0.99941](https://github.com/svpino/lunar-lander/blob/master/images/chart3.png)

Notice how the behavior of the agent is very erratic during the first three thousand iterations, but as it gets closer to the end, it becomes less intermittent because it starts exploiting what it has learned.

![Reward obtained when testing the agent using the learned weights during 100 episodes — Using α=.0001, γ=.99, and ε = 0.99941](https://github.com/svpino/lunar-lander/blob/master/images/chart4.png)

Finally, the above chart shows the results obtained by the agent after training is complete. In this case, the agent was run three times, and the individual episode reward was plotted. Notice how no crashes happened during the 100 episodes for any of the runs indicating the training process was successful and our agent solved the environment. The average reward for each run, as shown in the legend of the chart, was 209, 213, and 216.

## Conclusions
The primary challenge to train a successful agent to solve the Lunar Lander environment was the time constraint imposed by the schedule of the project, aggravated by the slow training time required by the algorithm to achieve good results. This didn't allow for a more exhaustive exploration of hyperparameters to get better convergence results, and opens the door to the following list of possible improvements and areas of research:

* How frequently we train the DNN could lead to significant speed improvements. Currently, the algorithm trains it after every time step to increase accuracy in detriment of performance.
* Training is currently done using a fixed learning rate, but decaying it over time may lead to better results.
* The initial learning rate and discount factors should benefit from a more exhaustive grid search beyond the nine combination of values explored.
* The current algorithm uses a ε-greedy strategy, but other approaches may lead to better results.
* The use of a Double Q-learning Network (DDQN) instead of a simpler DQN may increase the accuracy of the agent.
* The use of a dynamic-sized replay buffer might show improvements as shown in (Liu et al., 2017) 

Besides having a lot of room for improvement, the current implementation solved the environment successfully and showed the importance of a proper selection of hyperparameter to get the agent to converge. This report focused specifically on the influence of learning rate, discount factor, and ε-decay and how they affected the agent behavior. 

The use of DQN with two parallel networks as suggested in (Lillicrap et al., 2015) proved to be an efficient way to tackle the infinite state space of this problem.

## References
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. _Human-level Control Through Deep Reinforcement Learning._ (2015). Nature, 518(7540):529–533. http://dx.doi.org/10.1038/nature14236.

T. Lillicrap, J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. _Continuous Control With Deep Reinforcement Learning._ (2015). arXiv preprint arXiv:1509.02971.

R. Liu, J. Zou. _The Effects of Memory Replay in Reinforcement Learning._ (2017). arXiv:1710.06574.



