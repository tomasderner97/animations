from itertools import count
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.misc import derivative


def get_retarded_time_secant(x,
                             z,
                             q_x_func,
                             q_z_func,
                             t,
                             C):
    print("computing retarded time")

    def func(t_r):
        return np.sqrt((x - q_x_func(t_r)) ** 2 + (z - q_z_func(t_r)) ** 2) - C * (t - t_r)

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

    # indices = np.where(np.logical_not(np.isnan(res_n_m_1)))
    # sol[indices] = res_n_m_1[indices]

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

    t_start = time()
    t_ret = get_retarded_time_secant(x, z, q_x_func, q_z_func, t, C)
    print("time to compute t_ret:", time() - t_start)
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


def E_theta(x,
            z,
            t,
            q_x_func=None,
            q_z_func=None,
            C=0,
            EPS_0=0,
            Q=0):
    theta = np.arctan2(abs(x), z)
    e_theta_x = np.cos(theta)
    e_theta_z = -np.sin(theta)

    E_x, E_z = E_at_point(x, z, t, q_x_func, q_z_func, C, EPS_0, Q)

    return E_x * e_theta_x * np.sign(x) + E_z * e_theta_z


def double_sqrt(x):
    return np.sqrt(abs(x)) * np.sign(x)


counter = count()


def draw_dipole_field(t):
    halfside = 60
    T = 6

    def z_func(t_):
        return np.sin(2 * np.pi * t_ / T)

    x, z = np.mgrid[-halfside:halfside:200j, -halfside:halfside:200j]

    t_start = time()
    positive_field = E_theta(x, z + 2, t,
                             q_z_func=z_func)
    negative_field = E_theta(x, z - 2, t + T / 2,
                             q_z_func=z_func)
    print("time:", time() - t_start)
    field = double_sqrt(positive_field - negative_field)

    plt.clf()
    plt.axis("off")
    plt.imshow(field.T, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower",
               # vmin=double_sqrt(-0.0332992), vmax=double_sqrt(0.0299505),
               vmin=double_sqrt(-0.03), vmax=double_sqrt(0.025),
               interpolation="quadric", )
    # cmap="inferno")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    plt.plot([0], [z_func(t) - 2], "ro")
    # print(pos_z(t), field[25,25])
    plt.plot([0], [z_func(t + T / 2) + 2], "go")
    # plt.savefig(f"out3/{str(counter).rjust(3, '0')}-t={t:.2g}.png", bbox_inches="tight")
    print(f"done t = {t}")
    print(next(counter))


anim = FuncAnimation(plt.gcf(), draw_dipole_field, np.arange(0, 12, 0.1), interval=100)
plt.show()