import gym
from gym import wrappers
import numpy as np
import random, tempfile, os
from collections import deque
import tensorflow as tf
import time

TRAINING = False

LEARNING_RATE = [0.01, 0.001, 0.0001]
DISCOUNT_FACTOR = [0.9, 0.99, 0.999]

"""
Here are the values of this constant in order to achieve a proper balance of exploitation versus exploration 
at 5,000 episodes:

* 0.99910 - 99.99% exploitation + 0.01% exploration
* 0.99941 - 99.95% exploitation + 0.05% exploration
* 0.99954 - 99.90% exploitation + 0.10% exploration
* 0.99973 - 99.75% exploitation + 0.25% exploration
* 0.99987 - 99.50% exploitation + 0.50% exploration
"""
EPSILON_DECAY = [0.99910, 0.99941, 0.99954, 0.99973, 0.99987]

LEARNING_EPISODES = 5000
TESTING_EPISODES = 100
REPLAY_BUFFER_SIZE = 250000
REPLAY_BUFFER_BATCH_SIZE = 32
MINIMUM_REWARD = -250
STATE_SIZE = 8
NUMBER_OF_ACTIONS = 4
WEIGHTS_FILENAME = './weights/weights.h5'

class Agent:
	def __init__(self, training, learning_rate, discount_factor, epsilon_decay):
		self.training = training
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon_decay = epsilon_decay
		self.epsilon = 1.0 if self.training else 0.0
		self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

		self._create_networks()

		self.saver = tf.train.Saver()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		if not training:
			self._load_weights()

	def choose_action(self, s):
		if not self.training or np.random.rand() > self.epsilon:
			return np.argmax(self._Q(np.reshape(s, [1, STATE_SIZE]))[0])

		return np.random.choice(NUMBER_OF_ACTIONS)

	def store(self, s, a, r, s_, is_terminal):
		if self.training:
			self.replay_buffer.append((np.reshape(s, [1, STATE_SIZE]), a, r, np.reshape(s_, [1, STATE_SIZE]), is_terminal))

	def optimize(self, s, a, r, s_, is_terminal):
		if self.training and len(self.replay_buffer) > REPLAY_BUFFER_BATCH_SIZE:
			batch = np.array(random.sample(list(self.replay_buffer), REPLAY_BUFFER_BATCH_SIZE))
			s = np.vstack(batch[:, 0])
			a = np.array(batch[:, 1], dtype=int)
			r = np.array(batch[:, 2], dtype=float)
			s_ = np.vstack(batch[:, 3])

			non_terminal_states = np.where(batch[:, 4] == False)

			if len(non_terminal_states[0]) > 0:
				a_ = np.argmax(self._Q(s_)[non_terminal_states, :][0], axis=1)
				r[non_terminal_states] += np.multiply(self.discount_factor, self._Q_target(s_)[non_terminal_states, a_][0])

			y = self._Q(s)
			y[range(REPLAY_BUFFER_BATCH_SIZE), a] = r
			self._optimize(s, y)

	def close(self):
		if self.training:
			print("Saving agent weights to disk...")
			save_path = self.saver.save(self.sess, WEIGHTS_FILENAME)

	def update(self): 
		if self.training:
			Q_W1, Q_W2, Q_W3, Q_b1, Q_b2, Q_b3 = self._get_variables("Q")
			Q_target_W1, Q_target_W2, Q_target_W3, Q_target_b1, Q_target_b2, Q_target_b3 = self._get_variables("Q_target")
			self.sess.run([Q_target_W1.assign(Q_W1), Q_target_W2.assign(Q_W2), Q_target_W3.assign(Q_W3), Q_target_b1.assign(Q_b1), Q_target_b2.assign(Q_b2), Q_target_b3.assign(Q_b3)])

		if self.epsilon > 0.01:
			self.epsilon *= self.epsilon_decay

	def _load_weights(self):
		print("Loading agent weights from disk...")
		try:
			self.saver.restore(self.sess, WEIGHTS_FILENAME)
		except Exception as e:
			print("Error loading agent weights from disk.", e)

	def _optimize(self, s, y):
		optimizer, loss, Q_network = self.sess.run([self.optimizer, self.loss, self.Q_network], {self.Q_X: s, self.Q_y: y})

	def _Q(self, s):
		return self.sess.run(self.Q_network, {self.Q_X: s})

	def _Q_target(self, s):
		return self.sess.run(self.Q_target_network, {self.Q_target_X: s})

	def _create_networks(self):
		with tf.variable_scope("Q", reuse=tf.AUTO_REUSE):
			self.Q_X, self.Q_network = self._create_network()
			self.Q_y = tf.placeholder(shape=[None, NUMBER_OF_ACTIONS], dtype=tf.float32, name="y")

		with tf.name_scope("loss"):
			self.loss = tf.reduce_mean(tf.squared_difference(self.Q_y, self.Q_network))

		with tf.name_scope("train"):
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		with tf.variable_scope("Q_target"):
			self.Q_target_X, self.Q_target_network = self._create_network()

	def _create_network(self):
		X = tf.placeholder(shape=[None, STATE_SIZE], dtype=tf.float32, name="X")

		layer1 = tf.contrib.layers.fully_connected(X, 32, activation_fn=tf.nn.relu)
		layer2 = tf.contrib.layers.fully_connected(layer1, 32, activation_fn=tf.nn.relu)
		network = tf.contrib.layers.fully_connected(layer2, NUMBER_OF_ACTIONS, activation_fn=None)

		return X, network

	def _get_variables(self, scope):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			W1 = tf.get_variable("fully_connected/weights")
			W2 = tf.get_variable("fully_connected_1/weights")
			W3 = tf.get_variable("fully_connected_2/weights")
			b1 = tf.get_variable("fully_connected/biases")
			b2 = tf.get_variable("fully_connected_1/biases")
			b3 = tf.get_variable("fully_connected_2/biases")

		return W1, W2, W3, b1, b2, b3

if __name__ == "__main__":
	np.set_printoptions(precision=2)

	env = gym.make("LunarLander-v2")
	average_reward = deque(maxlen=100)

	agent = Agent(TRAINING, LEARNING_RATE[2], DISCOUNT_FACTOR[1], EPSILON_DECAY[1])

	print("Alpha: %.4f Gamma: %.3f Epsilon %.5f" % (agent.learning_rate, agent.discount_factor, agent.epsilon_decay))
	
	for episode in range(LEARNING_EPISODES if TRAINING else TESTING_EPISODES):
		current_reward = 0

		s = env.reset()

		for t in range(1000):
			if not TRAINING: 
				env.render()

			a = agent.choose_action(s)
			s_, r, is_terminal, info = env.step(a)

			current_reward += r

			agent.store(s, a, r, s_, is_terminal)
			agent.optimize(s, a, r, s_, is_terminal)

			s = s_

			if is_terminal or current_reward < MINIMUM_REWARD:
				break

		agent.update()

		average_reward.append(current_reward)

		print("%i, %.2f, %.2f, %.2f" % (episode, current_reward, np.average(average_reward), agent.epsilon))

	env.close()
	agent.close()
