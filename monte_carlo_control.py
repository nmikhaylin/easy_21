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

class MonteCarloControl(object):
  def __init__(self):
    self._q = dict()
    self._state_counts = dict()
    self._state_action_counts = dict()
    self._env = environment.Environment()

  def learn(self):
    for i in range(FLAGS.num_episodes):
      if i % 10000 == 1:
        print i
      self.run_episode()
    self.plot_q()

  def run_episode(self):
    current_state = self._env.generate_starting_state()
    total_reward = 0.0
    actions = []
    while not current_state.is_terminal:
      next_action = self.generate_action(current_state)
      self._state_action_counts[(current_state,next_action)] = (
          self._state_action_counts.get((current_state, next_action), 0) + 1)
      self._state_counts[current_state] = self._state_counts.get(
          current_state, 0) + 1
      actions.append((current_state, next_action))
      current_state, reward = self._env.step(current_state, next_action)
      total_reward += reward
    for (state, action) in actions:
      state_actions = self._q.setdefault(state, dict())
      previous_q = state_actions.get(action, 0.0)
      state_actions[action] = previous_q + (
          1.0 / float(self._state_action_counts[(state, action)]) *
          (total_reward - previous_q))

  def get_explore_threshold(self, state):
    return float(FLAGS.N_0) / (
        float(FLAGS.N_0) + float(self._state_counts.setdefault(state, 0)))

  def plot_q(self):
    (x,y,z) = ([],[],[])
    for (state, action_dict) in self._q.items():
      x.append(float(state.dealer_card.get_game_value()))
      y.append(float(state.player_sum))
      _, value = max(action_dict.items(), key=lambda x: x[1])
      z.append(value)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(np.array(x),np.array(y),np.array(z),
               linewidth=1, antialiased=False)
    plt.show()

  def generate_action(self, state):
    if (random.random() < self.get_explore_threshold(state) or
        state not in self._q):
      return random.choice(environment.ACTIONS)
    action_values = self._q.setdefault(state, dict())
    best_action, value = max(action_values.items(), key=lambda x: x[1])
    return best_action



def main(unused_argv):
  control = MonteCarloControl()
  control.learn()


if __name__ == "__main__":
  tf.app.run()
