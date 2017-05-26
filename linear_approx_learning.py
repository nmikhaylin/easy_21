import random
import environment
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", 10, "Number of episodes to run.")

flags.DEFINE_integer("N_0", 100, "Epsilon greedy constant.")

flags.DEFINE_float("td_lambda", .1, "Lambda for TD(lambda)")

flags.DEFINE_float("gamma", 1.0, "Gamma representing time decay.")

FEATURE_SIZE = 36



class LinearApproxLearning(object):
  def __init__(self):
    self._state_counts = dict()
    self._env = environment.Environment()
    self._weights = np.array([0.0] * FEATURE_SIZE)

  def learn(self):
    for i in range(FLAGS.num_episodes):
      if i % 10000 == 1:
        print i
      self.run_episode()
    self.plot_q()

  def get_alpha(self):
    return 0.01

  def update_model(self, state, action, update_magnitude):
    self._weights += (self.generate_feature_vector(
        state, action) * (self.get_alpha() * update_magnitude))

  def run_episode(self):
    initial_state = self._env.generate_starting_state()
    next_action = self.generate_action(initial_state)
    traces = dict()
    mult = 1.0
    while not initial_state.is_terminal:
      self._state_counts[initial_state] = self._state_counts.get(
          initial_state, 0) + 1
      traces[(initial_state, next_action)] = traces.get(
          (initial_state, next_action), 0) + 1 / mult
      prev_action = next_action
      next_state, reward = self._env.step(initial_state, next_action)

      action_q = self.evaluate_model(initial_state, prev_action)
      q_prime = 0.0
      if not next_state.is_terminal:
        next_action = self.generate_action(next_state)
        # Update
        q_prime = self.evaluate_model(next_state, next_action)
      update = reward + q_prime - action_q
      for state,action in traces.keys():
        self.update_model(state, action,
                          update * traces[(state,action)] * mult)
      initial_state = next_state
      mult *= FLAGS.td_lambda

  def get_explore_threshold(self, state):
    return .05

  def plot_q(self):
    (x,y,z) = ([],[],[])
    for d in range(1,10):
      for p in range(1,21):
        state = environment.State(environment.Card(
          environment.COLOR_BLACK, d), p)
        x.append(float(d))
        y.append(float(p))
        value = max(self.evaluate_model(state, environment.ACTION_HIT),
                       self.evaluate_model(state, environment.ACTION_STICK))
        z.append(value)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(np.array(x),np.array(y),np.array(z),
               linewidth=1, antialiased=False)
    plt.show()

  @staticmethod
  def generate_feature_vector(state, action):
    dealer_feat = []
    player_feat = []
    action_feat = None
    if 1 <= state.dealer_card.value <= 4:
      dealer_feat.append(0)
    if 4 <= state.dealer_card.value <= 7:
      dealer_feat.append(1)
    if 7 <= state.dealer_card.value <= 10:
      dealer_feat.append(2)
    if 1 <= state.player_sum <= 6:
      player_feat.append(0)
    if 4 <= state.player_sum <= 9:
      player_feat.append(1)
    if 7 <= state.player_sum <= 12:
      player_feat.append(2)
    if 10 <= state.player_sum <= 15:
      player_feat.append(3)
    if 13 <= state.player_sum <= 18:
      player_feat.append(4)
    if 16 <= state.player_sum <= 21:
      player_feat.append(5)
    if action == environment.ACTION_HIT:
      action_feat = 0
    if action == environment.ACTION_STICK:
      action_feat = 1
    output_feature = np.array([0] * FEATURE_SIZE)
    for d in dealer_feat:
      for p in player_feat:
        index = d * 2 * 6 + p * 2 + action_feat
        output_feature[index] = 1
    return output_feature

  def generate_action(self, state):
    if random.random() < self.get_explore_threshold(state):
      return random.choice(environment.ACTIONS)
    hit_val = self.evaluate_model(state, environment.ACTION_HIT)
    stick_val = self.evaluate_model(state, environment.ACTION_STICK)
    if hit_val > stick_val:
      return environment.ACTION_HIT
    return environment.ACTION_STICK

  def evaluate_model(self, state, action):
    return np.dot(self.generate_feature_vector(state, action), self._weights)

def main(unused_argv):
  control = LinearApproxLearning()
  control.learn()


if __name__ == "__main__":
  tf.app.run()
