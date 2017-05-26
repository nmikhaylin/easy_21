import unittest
from mock import MagicMock
import td_learning
from environment import *

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

SAMPLE_STATE = State(Card(COLOR_BLACK, 10), 11)

class TestTDLearningControl(unittest.TestCase):
  def test_init(self):
    self.assertTrue(td_learning.TDLearningControl())

  def test_generate_action(self):
    tdlc  = td_learning.TDLearningControl()
    tdlc.get_explore_threshold = MagicMock(return_value=0.0)

    tdlc._q[SAMPLE_STATE] = {ACTION_HIT: 1.0, ACTION_STICK:2.0}
    self.assertEqual(ACTION_STICK, tdlc.generate_action(SAMPLE_STATE))

    tdlc._q[SAMPLE_STATE] = {ACTION_HIT: 2.0, ACTION_STICK:1.0}
    self.assertEqual(ACTION_HIT, tdlc.generate_action(SAMPLE_STATE))

    tdlc._q[SAMPLE_STATE] = {ACTION_STICK:1.0}
    self.assertEqual(ACTION_STICK, tdlc.generate_action(SAMPLE_STATE))

    del tdlc._q[SAMPLE_STATE]
    self.assertIn(tdlc.generate_action(SAMPLE_STATE), [ACTION_HIT, ACTION_STICK])

    tdlc.get_explore_threshold = MagicMock(return_value=1.0)
    self.assertIn(tdlc.generate_action(SAMPLE_STATE), [ACTION_HIT, ACTION_STICK])


  def test_get_explore_threshold(self):
    FLAGS.N_0 = 100
    tdlc  = td_learning.TDLearningControl()
    self.assertLessEqual(0.0, tdlc.get_explore_threshold(SAMPLE_STATE))
    self.assertGreaterEqual(1.0, tdlc.get_explore_threshold(SAMPLE_STATE))

    tdlc._state_counts[SAMPLE_STATE] = 100
    self.assertLessEqual(0.0, tdlc.get_explore_threshold(SAMPLE_STATE))
    self.assertGreaterEqual(1.0, tdlc.get_explore_threshold(SAMPLE_STATE))

    tdlc._state_counts[SAMPLE_STATE] = 10000
    self.assertLessEqual(0.0, tdlc.get_explore_threshold(SAMPLE_STATE))
    self.assertGreaterEqual(1.0, tdlc.get_explore_threshold(SAMPLE_STATE))


    tdlc._state_counts[SAMPLE_STATE] = 10000
    first_explore = tdlc.get_explore_threshold(SAMPLE_STATE)
    tdlc._state_counts[SAMPLE_STATE] = 1000000
    self.assertLess(tdlc.get_explore_threshold(SAMPLE_STATE), first_explore)

  def test_run_episode(self):
    tdlc  = td_learning.TDLearningControl()
    tdlc._env.generate_starting_state = MagicMock(
        return_value=State(Card(COLOR_BLACK, 10), 5))
    tdlc.generate_action = MagicMock(return_value=ACTION_HIT)
    tdlc._env.step = MagicMock(side_effect=[
      (State(Card(COLOR_BLACK, 10), 15),0),
      (State(Card(COLOR_BLACK, 10), 5, is_terminal=True),1),
      ])
    tdlc.run_episode()
    self.assertEqual(1, tdlc._state_counts[State(Card(COLOR_BLACK, 10), 5)])
    self.assertEqual(1, tdlc._state_action_counts[
      (State(Card(COLOR_BLACK, 10), 5), ACTION_HIT)])
    self.assertLess(
        0.0, tdlc._q[State(Card(COLOR_BLACK, 10), 5)][ACTION_HIT])
    self.assertLess(
        0.0, tdlc._q[State(Card(COLOR_BLACK, 10), 15)][ACTION_HIT])
    self.assertLess(
        tdlc._q[State(Card(COLOR_BLACK, 10), 5)][ACTION_HIT],
        tdlc._q[State(Card(COLOR_BLACK, 10), 15)][ACTION_HIT])
    print tdlc._q[State(Card(COLOR_BLACK, 10), 5)][ACTION_HIT]
    print tdlc._q[State(Card(COLOR_BLACK, 10), 15)][ACTION_HIT]


if __name__ == "__main__":
  unittest.main()
