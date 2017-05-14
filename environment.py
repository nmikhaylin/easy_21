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

  def get_game_value(self):
    return self.value if self.color == COLOR_BLACK else -self.value

  def __eq__(self, other):
    return self.color == other.color and self.value == other.value

  def __hash__(self):
    return hash((COLORS.index(self.color), self.value))

  def __str__(self):
    return "color:%s value:%d" % (self.color, self.value)


class State(object):
  def __init__(self, dealer_card, player_sum, is_terminal=False):
    self.dealer_card = dealer_card
    self.player_sum = player_sum
    self.is_terminal = is_terminal

  def __eq__(self, other):
    return (self.is_terminal == other.is_terminal and
            self.dealer_card == other.dealer_card and
            self.player_sum == other.player_sum)

  def __hash__(self):
    return hash((self.dealer_card, self.player_sum, self.is_terminal))

  def __str__(self):
    return "dealer: %s player: %d terminal: %s" % (
        str(self.dealer_card), self.player_sum, self.is_terminal)


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
    new_card = self._generate_random_card()
    if action == ACTION_HIT:
      new_player_value = state.player_sum + new_card.get_game_value()
      if new_player_value < 1 or new_player_value > 21:
        return TERMINAL_STATE, -1
      return State(state.dealer_card, new_player_value), 0.0
    current_dealer_value = state.dealer_card.get_game_value()
    # STICK
    current_dealer_value += new_card.get_game_value()
    while current_dealer_value < 17:
      if current_dealer_value < 1:
        return TERMINAL_STATE, 1
      drawn_card = self._generate_random_card()
      current_dealer_value += drawn_card.get_game_value()
    if current_dealer_value > 21:
      return TERMINAL_STATE, 1
    if state.player_sum > current_dealer_value:
      return TERMINAL_STATE, 1
    if state.player_sum == current_dealer_value:
      return TERMINAL_STATE, 0
    return TERMINAL_STATE, -1

  def generate_starting_state(self):
    return State(self._generate_random_card(force_black=True),
                 self._generate_random_card(force_black=True).get_game_value())


  @staticmethod
  def _generate_random_card(force_black=False):
    return Card(COLOR_BLACK if force_black else random.choice(COLORS),
                random.randint(1,10))
