import environment
from environment import *

import unittest
from mock import MagicMock

class TestEnvironment(unittest.TestCase):
  def test_init(self):
    self.assertTrue(Environment())

  def test_generate_random_card(self):
    (red_count, black_count) = (0,0)
    value_counts = [0]*10
    for i in range(100):
      card = Environment._generate_random_card()
      self.assertIn(card.color, COLORS)
      self.assertIn(card.value, range(1, 11))
      if card.color == COLOR_RED:
        red_count += 1
      else:
        black_count += 1
      value_counts[card.value - 1] += 1
    print "red_count: %d \n black_count: %d \n value_counts:%s" % (
        red_count, black_count, value_counts)

  def test_generate_card_force_black(self):
    for i in range(100):
      card = Environment._generate_random_card(force_black=True)
      self.assertEqual(COLOR_BLACK, card.color)

  def test_dict(self):
    d = dict()
    d[State(Card(COLOR_BLACK, 5), 5)] = 1
    self.assertFalse(State(Card(COLOR_RED, 5), 5) in d)
    self.assertTrue(State(Card(COLOR_BLACK, 5), 5) in d)

  def test_step_hit(self):
    env = Environment()
    starting_state = State(Card(COLOR_BLACK, 5), 6)
    env._generate_random_card = MagicMock(side_effect=[Card(COLOR_RED, 4)])
    (new_state, reward) = env.step(starting_state, ACTION_HIT)
    self.assertFalse(new_state.is_terminal)
    self.assertEqual(2, new_state.player_sum)
    self.assertEqual(0.0, reward)
    self.assertEqual(Card(COLOR_BLACK, 5), new_state.dealer_card)

    starting_state = State(Card(COLOR_BLACK, 5), 6)
    env._generate_random_card = MagicMock(side_effect=[Card(COLOR_BLACK, 4)])
    (new_state, reward) = env.step(starting_state, ACTION_HIT)
    self.assertFalse(new_state.is_terminal)
    self.assertEqual(0.0, reward)
    self.assertEqual(10, new_state.player_sum)
    self.assertEqual(Card(COLOR_BLACK, 5), new_state.dealer_card)

    starting_state = State(Card(COLOR_BLACK, 5), 14)
    env._generate_random_card = MagicMock(side_effect=[Card(COLOR_BLACK, 5)])
    (new_state, reward) = env.step(starting_state, ACTION_HIT)
    self.assertFalse(new_state.is_terminal)
    self.assertEqual(0.0, reward)
    self.assertEqual(19, new_state.player_sum)
    self.assertEqual(Card(COLOR_BLACK, 5), new_state.dealer_card)

    starting_state = State(Card(COLOR_BLACK, 5), 17)
    env._generate_random_card = MagicMock(side_effect=[Card(COLOR_BLACK, 5)])
    (new_state, reward) = env.step(starting_state, ACTION_HIT)
    self.assertTrue(new_state.is_terminal)
    self.assertEqual(-1, reward)

    starting_state = State(Card(COLOR_BLACK, 5), 4)
    env._generate_random_card = MagicMock(side_effect=[Card(COLOR_RED, 5)])
    (new_state, reward) = env.step(starting_state, ACTION_HIT)
    self.assertTrue(new_state.is_terminal)
    self.assertEqual(-1, reward)

  def test_step_stick(self):
    env = Environment()
    starting_state = State(Card(COLOR_BLACK, 5), 6)
    env._generate_random_card = MagicMock(side_effect=[
      Card(COLOR_RED, 4), Card(COLOR_RED, 4)])
    (new_state, reward) = env.step(starting_state, ACTION_STICK)
    self.assertTrue(new_state.is_terminal)
    self.assertEqual(1.0, reward)

    starting_state = State(Card(COLOR_BLACK, 5), 6)
    env._generate_random_card = MagicMock(side_effect=[
      Card(COLOR_BLACK, 4), Card(COLOR_RED, 4), Card(COLOR_BLACK, 6),
      Card(COLOR_BLACK, 7)])
    (new_state, reward) = env.step(starting_state, ACTION_STICK)
    self.assertTrue(new_state.is_terminal)
    self.assertEqual(-1.0, reward)

    starting_state = State(Card(COLOR_BLACK, 5), 18)
    env._generate_random_card = MagicMock(side_effect=[
      Card(COLOR_BLACK, 4), Card(COLOR_RED, 4), Card(COLOR_BLACK, 6),
      Card(COLOR_BLACK, 7)])
    (new_state, reward) = env.step(starting_state, ACTION_STICK)
    self.assertTrue(new_state.is_terminal)
    self.assertEqual(0.0, reward)

    starting_state = State(Card(COLOR_BLACK, 5), 18)
    env._generate_random_card = MagicMock(side_effect=[
      Card(COLOR_BLACK, 4), Card(COLOR_RED, 4), Card(COLOR_BLACK, 10),
      Card(COLOR_BLACK, 7)])
    (new_state, reward) = env.step(starting_state, ACTION_STICK)
    self.assertTrue(new_state.is_terminal)
    self.assertEqual(1.0, reward)

  def test_generate_starting_state(self):
    env = Environment()
    for i in range(100):
      state = env.generate_starting_state()
      self.assertEqual(COLOR_BLACK, state.dealer_card.color)
      self.assertLessEqual(1, state.player_sum)
      self.assertLessEqual(1 , state.dealer_card.get_game_value())


class TestCard(unittest.TestCase):
  def test_init(self):
    self.assertTrue(Card(COLOR_RED, 1))
    self.assertTrue(Card(COLOR_BLACK, 5))

  def test_hash(self):
    self.assertEqual(hash(Card(COLOR_RED, 10)), hash(Card(COLOR_RED, 10)))
    self.assertNotEqual(hash(Card(COLOR_RED, 9)), hash(Card(COLOR_RED, 10)))
    self.assertNotEqual(hash(Card(COLOR_BLACK, 10)), hash(Card(COLOR_RED, 10)))

  def test_card_value(self):
    self.assertEqual(1, Card(COLOR_BLACK, 1).get_game_value())
    self.assertEqual(5, Card(COLOR_BLACK, 5).get_game_value())
    self.assertEqual(-1, Card(COLOR_RED, 1).get_game_value())
    self.assertEqual(-5, Card(COLOR_RED, 5).get_game_value())


class TestState(unittest.TestCase):
  def test_init(self):
    self.assertTrue(State(Card(COLOR_RED, 5), 12))
    self.assertTrue(State(Card(COLOR_RED, 5), 22, True).is_terminal)

  def test_hash(self):
    self.assertEqual(hash(State(Card(COLOR_RED, 10), 10)),
                     hash(State(Card(COLOR_RED, 10), 10)))
    self.assertNotEqual(hash(State(Card(COLOR_BLACK, 10), 10)),
                        hash(State(Card(COLOR_RED, 10), 10)))
    self.assertNotEqual(hash(State(Card(COLOR_RED, 10), 11)),
                        hash(State(Card(COLOR_RED, 10), 10)))


if __name__ == "__main__":
  unittest.main()
