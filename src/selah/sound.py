import math

SPEED_OF_SOUND = 343.0


def db(gain: float) -> float:
    """
    Converts a raw gain into decibels
    """
    return 10 * math.log10(gain)


def from_db(gain: float) -> float:
    """
    Converts from decibels to raw gain
    """
    return math.pow(10, gain / 10)
