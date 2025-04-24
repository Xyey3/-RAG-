from .linear import Notears
from .low_rank import NotearsLowRank

from ncastle.backend import backend

if backend == 'pytorch':
    from .torch import NotearsNonlinear
    from .torch import GOLEM
