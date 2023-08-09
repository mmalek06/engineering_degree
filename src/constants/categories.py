from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(self, start: int, count: int, last_values) -> str:
        return str(self).lower()


class Categories(AutoName):
    AKIEC = auto()
    BCC = auto()
    BKL = auto()
    DF = auto()
    MEL = auto()
    NV = auto()
    SCC = auto()
    SEK = auto()
    VASC = auto()
