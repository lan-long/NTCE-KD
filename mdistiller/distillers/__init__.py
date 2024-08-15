from ._base import Vanilla
from .KD import KD
from .NTCE import NTCE 

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "NTCE": NTCE,
}
