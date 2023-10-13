import numpy as np


def comp2speed(u, v, w=None):
    if w is None:
        return np.sqrt(u ** 2 + v ** 2)
    else:
        return np.sqrt(u ** 2 + v ** 2 + w ** 2)


def comp2alpha(u, v, unit_output="radian"):
    alpha = np.where(u == 0,
                 np.where(v == 0, 0, np.sign(v) * np.pi / 2),
                 np.arctan(v / u))

    if unit_output == "degree":
        alpha = np.rad2deg(alpha)

    return alpha


def alpha2dir(wind_dir, alpha, unit_direction ="degree", unit_alpha="radian", unit_output="radian"):
    if unit_direction == "degree":
        wind_dir = np.deg2rad(wind_dir)

    if unit_alpha == "degree":
        alpha = np.deg2rad(alpha)

    dir = wind_dir - alpha

    if unit_output == "degree":
        dir = np.rad2deg(dir)

    return dir


def wind2comp(uv, dir, unit_direction):

    if unit_direction == "degree":
        dir = np.deg2rad(dir)

    u = -np.sin(dir) * uv
    v = -np.cos(dir) * uv

    return u, v


def comp2dir(u, v, unit_output="degree"):
    if unit_output == "degree":
        return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)
    else:
        raise NotImplementedError
