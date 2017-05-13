import collections
import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

random.seed(42)

COLOR_RED = "R"
COLOR_BLACK = "B"
# twice as likely to generate a black card.
COLORS = [COLOR_RED, COLOR_BLACK, COLOR_BLACK]

ACTION_HIT = "H"
ACTION_STICK = "S"
ACTIONS = [ACTION_HIT, ACTION_STICK]




Card = collections.namedtuple("Card", ["color", "value"])

class Card(object):
  def __init__(self, color, value):
    self.color = color
    self.value = value

  def GetGameValue(self):
    return self.value if self.color == COLOR_BLACK else -self.value

  def __eq__(self, other):
    return self.color == other.color and self.value == other.value


class State(object):
  def __init__(self, dealer_card, player_sum, is_terminal=False):
    self.dealer_card = dealer_card
    self.player_sum = player_sum
    self.is_terminal = is_terminal

  def IsTerminal(self):
    return self.is_terminal

  def __eq__(self, other):
    return (self.is_terminal == other.is_terminal and
            self.dealer_card == other.dealer_card and
            self.player_sum == other.player_sum)


TERMINAL_STATE = State(Card(COLOR_RED, 1), 1, True)


class Environment(object):

  def __init__(self):
    pass

  def step(self, state, action):
    """ Gets a sample of the next state given a state and an action.

    Args:
      state: A State that represents the state before the action.
      action: One of ACTIONS representing the action taken by the player.

    Returns:
      new_state: The state after the resolution of the action, can be terminal.
      reward: The sum of the rewards encountered while resolving action.
    """
    reward = 0.0
    new_card = self._GenerateRandomCard()
    if action == ACTION_HIT:
      new_player_value = state.player_sum + new_card.GetGameValue()
      if new_player_value < 1 or new_player_value > 21:
        return TERMINAL_STATE, -1
      return State(state.dealer_card, new_player_value), 0.0
    current_dealer_value = state.dealer_card.GetGameValue()
    # STICK
    current_dealer_value += new_card.GetGameValue()
    while current_dealer_value < 17:
      if current_dealer_value < 1:
        return TERMINAL_STATE, 1
      drawn_card = self._GenerateRandomCard()
      current_dealer_value += drawn_card.GetGameValue()
    if current_dealer_value > 21:
      return TERMINAL_STATE, 1
    if state.player_sum > current_dealer_value:
      return TERMINAL_STATE, 1
    if state.player_sum == current_dealer_value:
      return TERMINAL_STATE, 0
    return TERMINAL_STATE, -1

  @staticmethod
  def _GenerateRandomCard():
    return Card(random.choice(COLORS), random.randint(1,10))

