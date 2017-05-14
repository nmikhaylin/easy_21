import unittest
from mock import MagicMock
import monte_carlo_control
from environment import *

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

SAMPLE_STATE = State(Card(COLOR_BLACK, 10), 11)

class TestMonteCarloControl(unittest.TestCase):
  def test_init(self):
    self.assertTrue(monte_carlo_control.MonteCarloControl())

  def test_generate_action(self):
    mcc  = monte_carlo_control.MonteCarloControl()
    mcc.get_explore_threshold = MagicMock(return_value=0.0)

    mcc._q[SAMPLE_STATE] = {ACTION_HIT: 1.0, ACTION_STICK:2.0}
    self.assertEqual(ACTION_STICK, mcc.generate_action(SAMPLE_STATE))

    mcc._q[SAMPLE_STATE] = {ACTION_HIT: 2.0, ACTION_STICK:1.0}
    self.assertEqual(ACTION_HIT, mcc.generate_action(SAMPLE_STATE))

    mcc._q[SAMPLE_STATE] = {ACTION_STICK:1.0}
    self.assertEqual(ACTION_STICK, mcc.generate_action(SAMPLE_STATE))

    del mcc._q[SAMPLE_STATE]
    self.assertIn(mcc.generate_action(SAMPLE_STATE), [ACTION_HIT, ACTION_STICK])

    mcc.get_explore_threshold = MagicMock(return_value=1.0)
    self.assertIn(mcc.generate_action(SAMPLE_STATE), [ACTION_HIT, ACTION_STICK])


  def test_get_explore_threshold(self):
    FLAGS.N_0 = 100
    mcc  = monte_carlo_control.MonteCarloControl()
    self.assertLessEqual(0.0, mcc.get_explore_threshold(SAMPLE_STATE))
    self.assertGreaterEqual(1.0, mcc.get_explore_threshold(SAMPLE_STATE))

    mcc._state_counts[SAMPLE_STATE] = 100
    self.assertLessEqual(0.0, mcc.get_explore_threshold(SAMPLE_STATE))
    self.assertGreaterEqual(1.0, mcc.get_explore_threshold(SAMPLE_STATE))

    mcc._state_counts[SAMPLE_STATE] = 10000
    self.assertLessEqual(0.0, mcc.get_explore_threshold(SAMPLE_STATE))
    self.assertGreaterEqual(1.0, mcc.get_explore_threshold(SAMPLE_STATE))


    mcc._state_counts[SAMPLE_STATE] = 10000
    first_explore = mcc.get_explore_threshold(SAMPLE_STATE)
    mcc._state_counts[SAMPLE_STATE] = 1000000
    self.assertLess(mcc.get_explore_threshold(SAMPLE_STATE), first_explore)

  def test_run_episode(self):
    mcc  = monte_carlo_control.MonteCarloControl()
    mcc._env.generate_starting_state = MagicMock(
        return_value=State(Card(COLOR_BLACK, 10), 5))
    mcc.generate_action = MagicMock(return_value=ACTION_HIT)
    mcc._env.step = MagicMock(side_effect=[
      (State(Card(COLOR_BLACK, 10), 15),0),
      (State(Card(COLOR_BLACK, 10), 5, is_terminal=True),1),
      ])
    mcc.run_episode()
    self.assertEqual(1, mcc._state_counts[State(Card(COLOR_BLACK, 10), 5)])
    self.assertEqual(1, mcc._state_action_counts[
      (State(Card(COLOR_BLACK, 10), 5), ACTION_HIT)])
    self.assertLess(0.0, mcc._q[State(Card(COLOR_BLACK, 10), 5)][ACTION_HIT])
    self.assertLess(
        0.0, mcc._q[State(Card(COLOR_BLACK, 10), 15)][ACTION_HIT])


if __name__ == "__main__":
  unittest.main()
