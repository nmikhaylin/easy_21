from environment import *
from mock import MagicMock,patch
import linear_approx_learning
import numpy as np
import tensorflow as tf
import unittest

flags = tf.app.flags
FLAGS = flags.FLAGS

SAMPLE_STATE = State(Card(COLOR_BLACK, 10), 11)

class TestLinearApproxLearning(unittest.TestCase):
  def test_init(self):
    self.assertTrue(linear_approx_learning.LinearApproxLearning())

  def test_generate_action(self):
    lal  = linear_approx_learning.LinearApproxLearning()
    lal.get_explore_threshold = MagicMock(return_value=0.0)
    with patch.object(lal, "evaluate_model", new=lambda state,action :
        1.0 if action == ACTION_HIT else 0.5):
      self.assertEqual(ACTION_HIT, lal.generate_action(SAMPLE_STATE))

    with patch.object(lal, "evaluate_model", new=lambda state,action :
        .5 if action == ACTION_HIT else 1.0):
      self.assertEqual(ACTION_STICK, lal.generate_action(SAMPLE_STATE))

  def test_get_explore_threshold(self):
    lal  = linear_approx_learning.LinearApproxLearning()
    self.assertLessEqual(0.05, lal.get_explore_threshold(SAMPLE_STATE))

  def test_evaluate_model(self):
    lal  = linear_approx_learning.LinearApproxLearning()
    lal._weights = np.array([.5] + [0] * 35)
    self.assertEqual(.5, lal.evaluate_model(
        State(Card(COLOR_BLACK, 1), 1), ACTION_HIT))
    self.assertEqual(0.0, lal.evaluate_model(
        State(Card(COLOR_BLACK, 11), 1), ACTION_HIT))

  def test_genearate_features(self):
    gen_features = linear_approx_learning.LinearApproxLearning.generate_feature_vector

    expected_feat = np.array([0] * 36)
    expected_feat[0] = 1
    self.assertTrue(np.array_equal(expected_feat, gen_features(
        State(Card(COLOR_BLACK, 1), 1), ACTION_HIT)))

    expected_feat = np.array([0] * 36)
    expected_feat[1] = 1
    self.assertTrue(np.array_equal(expected_feat, gen_features(
        State(Card(COLOR_BLACK, 1), 1), ACTION_STICK)))

    expected_feat = np.array([0] * 36)
    expected_feat[0] = 1
    expected_feat[12] = 1
    self.assertTrue(np.array_equal(expected_feat, gen_features(
        State(Card(COLOR_BLACK, 4), 1), ACTION_HIT)))

    expected_feat = np.array([0] * 36)
    expected_feat[0] = 1
    expected_feat[12] = 1
    expected_feat[2] = 1
    expected_feat[14] = 1
    self.assertTrue(np.array_equal(expected_feat, gen_features(
        State(Card(COLOR_BLACK, 4), 5), ACTION_HIT)))

    expected_feat = np.array([0] * 36)
    expected_feat[35] = 1
    self.assertTrue(np.array_equal(expected_feat, gen_features(
        State(Card(COLOR_BLACK, 10), 21), ACTION_STICK)))

  def test_run_episode(self):
    lal  = linear_approx_learning.LinearApproxLearning()
    lal._env.generate_starting_state = MagicMock(
        return_value=State(Card(COLOR_BLACK, 1), 1))
    lal.generate_action = MagicMock(return_value=ACTION_HIT)
    lal._env.step = MagicMock(side_effect=[
      (State(Card(COLOR_BLACK, 1), 1),0),
      (State(Card(COLOR_BLACK, 1), 1, is_terminal=True),1),
      ])
    lal.run_episode()
    self.assertLess(0.0, lal.evaluate_model(
      State(Card(COLOR_BLACK, 1), 1), ACTION_HIT))

    lal  = linear_approx_learning.LinearApproxLearning()
    lal._env.generate_starting_state = MagicMock(
        return_value=State(Card(COLOR_BLACK, 1), 1))
    lal.generate_action = MagicMock(return_value=ACTION_HIT)
    lal._env.step = MagicMock(side_effect=[
      (State(Card(COLOR_BLACK, 1), 1),0),
      (State(Card(COLOR_BLACK, 1), 1, is_terminal=True),-1),
      ])
    lal.run_episode()
    self.assertGreater(0.0, lal.evaluate_model(
      State(Card(COLOR_BLACK, 1), 1), ACTION_HIT))

if __name__ == "__main__":
  unittest.main()
