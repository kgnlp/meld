from collections.abc import Callable

from meld.formats import Split

type SplitFilter = Callable[[Split], bool]
