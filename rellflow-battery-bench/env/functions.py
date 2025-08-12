from pathlib import Path
import numpy as np
import torch

C_IN_K = 273.15
KILO = 1000


def s_to_hour(x):
    return x / 3600


def days_to_s(x):
    return x * 24 * 3600


def hours_to_s(x):
    return x * 3600


def min_to_s(x):
    return x * 60


def times_kilo(x):
    return x * KILO


def through_kilo(x):
    return x / KILO


def k_to_c(x):
    return x - C_IN_K


def c_to_k(x):
    return x + C_IN_K


def j_to_kwh(x):
    return x / (60**2 * KILO)


def kwh_to_j(x):
    return x * 3.6e6


class JToKw:
    def __init__(self, min) -> None:
        self.min = min

    def __call__(self, x):
        return x / (self.min * 60 * KILO)


def byte_to_gb(x):
    return x / 1024**3


def cyclic_encode(value, trigonometric_fun, max_val):
    if False:
        x = value if value < max_val else max_val
    else:
        x = np.rint(value)
    trig = trigonometric_fun(2 * np.pi * x / max_val)
    return trig


def sin_encode_hour(value, max_val=24):
    return cyclic_encode(value, np.sin, max_val)


def cos_encode_hour(value, max_val=24):
    return cyclic_encode(value, np.cos, max_val)


def sin_encode_day(value, max_val=7):
    return cyclic_encode(value, np.sin, max_val)


def cos_encode_day(value, max_val=7):
    return cyclic_encode(value, np.cos, max_val)


def sin_encode_month(value, max_val=12):
    return cyclic_encode(value, np.sin, max_val)


def cos_encode_month(value, max_val=12):
    return cyclic_encode(value, np.cos, max_val)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_path(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"File {path} not found")
    return path
