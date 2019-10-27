import os
from itertools import count
from time import time

from gizeh import gizeh
from moviepy.editor import ImageSequenceClip

import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.misc import derivative
from scipy.optimize import root_scalar, root

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

plt.rcParams["toolbar"] = "None"

T = 6
RATIO = 16 / 9


def z_func(t_):
    return np.sin(2 * np.pi * t_ / T)


def get_retarded_time_secant(x,
                             z,
                             t,
                             C):
    print("computing retarded time")

    def func(t_r):
        return np.sqrt(x ** 2 + (z - z_func(t_r)) ** 2) - C * (t - t_r)

    res_n_m_2 = np.full_like(x, t)
    res_n_m_1 = res_n_m_2 + 1

    sol = np.full_like(x, np.nan)
    for i in range(30):
        remaining = np.count_nonzero(np.isnan(sol))
        print(f"    iteration {i}, remaining: {remaining}")
        if not remaining:
            break
        f_n_m_1 = func(res_n_m_1)
        denominator = f_n_m_1 - func(res_n_m_2)
        indices = np.where(np.isclose(denominator, 0))
        sol[indices] = res_n_m_1[indices]
        res_n_m_2, res_n_m_1 = res_n_m_1, res_n_m_1 - f_n_m_1 * (res_n_m_1 - res_n_m_2) / denominator

    return sol


def E_at_point(x,
               z,
               t,
               q_x_func=None,
               q_z_func=None,
               C=0,
               EPS_0=0,
               Q=0):
    """
    Počítá komponenty X a Z pole pohybujícího se náboje v bodě [x, z]
    Parameters
    ----------
    x : float
        Pole hodnot X mřížky
    z : float
        Pole hodnot Y mřížky
    t : float
        Čas
    q_x_func : callable
        Funkce souřadnice X bodového náboje v čase
    q_z_func : callable
        Funkce souřadnice Z bodového náboje v čase
    C : float
        Rychlost světla
    EPS_0 : float
        Permitivita vakua
    Q : float
        Velikost náboje

    Returns
    -------
    E_x : float
        Xová složka E
    E_z : float
        Zová složka E
    """
    if not q_x_func:
        q_x_func = lambda t: 0
    if not q_z_func:
        q_z_func = lambda t: 0
    if not C:
        C = 2
    if not EPS_0:
        EPS_0 = 1
    if not Q:
        Q = 1

    # t_start = time()
    t_ret = get_retarded_time_secant(x, z, t, C)
    # t_ret_old = get_retarded_time(x, z, t, C)
    # diff = t_ret - t_ret_old
    # plt.imshow(diff)
    # plt.show()
    # print("time to compute t_ret:", time() - t_start)
    nice_r_x = x - q_x_func(t_ret) * np.ones_like(x)
    nice_r_z = z - q_z_func(t_ret) * np.ones_like(z)

    nice_r_norm = np.sqrt(nice_r_x ** 2 + nice_r_z ** 2)

    v_x_ret = derivative(q_x_func, t_ret, dx=1e-6)
    v_z_ret = derivative(q_z_func, t_ret, dx=1e-6)
    a_x_ret = derivative(q_x_func, t_ret, dx=1e-6, n=2)
    a_z_ret = derivative(q_z_func, t_ret, dx=1e-6, n=2)

    u_x = C * nice_r_x / nice_r_norm - v_x_ret
    u_z = C * nice_r_z / nice_r_norm - v_z_ret

    nice_r_dot_u = nice_r_x * u_x + nice_r_z * u_z
    const = Q / (4 * np.pi * EPS_0)
    front = nice_r_norm / (nice_r_dot_u ** 3)

    radiation_term_x = -nice_r_z * (u_z * a_x_ret - u_x * a_z_ret)
    radiation_term_z = nice_r_x * (u_z * a_x_ret - u_x * a_z_ret)

    E_x = const * front * radiation_term_x
    E_z = const * front * radiation_term_z

    return E_x, E_z


counter = count()


def draw_dipole_field(t):
    halfside = 20
    x, z = np.mgrid[-halfside * RATIO:halfside * RATIO:64j, -halfside:halfside:36j]

    positive_field_x, positive_field_z = E_at_point(x, z+2, t, q_z_func=z_func)
    negative_field_x, negative_field_z = E_at_point(x, z-2, t+T/2, q_z_func=z_func)
    # print("time:", time() - t_start)

    field_x = positive_field_x - negative_field_x
    field_z = positive_field_z - negative_field_z

    field_x[np.where(x**2 + z**2 < 16)] = np.nan
    # field_z[np.where(x**2 + z**2 < 20)] = np.nan

    plt.clf()
    plt.axis("off")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    plt.quiver(x, z, field_x, field_z)
    plt.plot([0], [z_func(t) - 2], "ro")
    # print(pos_z(t), field[25,25])
    plt.plot([0], [z_func(t + T / 2) + 2], "go")
    plt.savefig(f"out_quiver/{str(next(counter)).rjust(3, '0')}-t={t:.2g}.png")
    print(f"done t = {t}")



plt.figure(figsize=(10.8*RATIO, 10.8), dpi=100)
anim = FuncAnimation(plt.gcf(), draw_dipole_field, np.arange(0, 12, 0.1), interval=100)
plt.show()
