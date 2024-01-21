from enum import Enum

class State(Enum):
    Takeoff= 1
    Flying = 2
    Landing = 3
    Landed = 4