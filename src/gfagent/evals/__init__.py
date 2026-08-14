from .autoplay import PLAYER_PROFILES, PLAYER_STYLES, AutoPlayer, Session, play
from .critic import (
    Mechanical,
    Review,
    Verdict,
    compare,
    compare_pair,
    mechanical,
    review,
    sign_test,
)

__all__ = [
    "play", "Session", "AutoPlayer", "PLAYER_PROFILES", "PLAYER_STYLES",
    "review", "Review", "mechanical", "Mechanical",
    "compare", "compare_pair", "Verdict", "sign_test",
]
