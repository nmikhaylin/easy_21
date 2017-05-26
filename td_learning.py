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



class TDLearningControl(object):
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

  def get_alpha(self, state, action):
    return 1.0 / float(self._state_action_counts[(state, action)])


  def run_episode(self):
    initial_state = self._env.generate_starting_state()
    next_action = self.generate_action(initial_state)
    traces = dict()
    mult = 1.0
    while not initial_state.is_terminal:
      self._state_action_counts[(initial_state,next_action)] = (
          self._state_action_counts.get((initial_state, next_action), 0) + 1)
      self._state_counts[initial_state] = self._state_counts.get(
          initial_state, 0) + 1
      traces[(initial_state, next_action)] = traces.get(
          (initial_state, next_action), 0) + 1 / mult
      prev_action = next_action
      next_state, reward = self._env.step(initial_state, next_action)
      state_actions = self._q.setdefault(initial_state, dict())
      action_q = state_actions.get(prev_action, 0.0)
      q_prime = 0.0
      if not next_state.is_terminal:
        next_action = self.generate_action(next_state)
        # Update
        q_prime = self._q.setdefault(next_state, dict()).get(next_action, 0.0)
      update = reward + q_prime - action_q
      for state,action in traces.keys():
        self._q.setdefault(state, dict())[action] = self._q.setdefault(
            state, dict()).get(action, 0) + self.get_alpha(
                state, action) * update * traces[(state,action)] * mult
      initial_state = next_state
      mult *= FLAGS.td_lambda

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
  control = TDLearningControl()
  control.learn()


if __name__ == "__main__":
  tf.app.run()
