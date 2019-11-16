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


def z_func_derivative(t_):
    return 2 * np.pi / T * np.cos(2 * np.pi * t_ / T)


@np.vectorize
def get_retarded_time(x,
                      z,
                      t,
                      C):
    def func(t_r):
        return np.sqrt(x ** 2 + (z - z_func(t_r)) ** 2) - C * (t - t_r)

    result = root_scalar(func, x0=t, x1=t + 1)

    return result.root


def get_retarded_time_newton(x,
                             z,
                             t,
                             C):
    def func(t_r):
        return np.sqrt(x ** 2 + (z - z_func(t_r)) ** 2) - C * (t - t_r)

    def func_prime(t_r):
        z_q = z_func(t_r)
        return -(z - z_q) * z_func_derivative(t_r) / \
               np.sqrt(x ** 2 + (z - z_q) ** 2) + C

    res = np.full_like(x, t)

    for i in range(20):
        res -= func(res) / func_prime(res)

    return res


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


counter = count()


def draw_dipole_field(t):
    halfside = 60
    x, z = np.mgrid[-halfside * RATIO:halfside * RATIO:1920j, -halfside:halfside:1080j]

    # t_start = time()
    positive_field = E_theta(x, z + 2, t,
                             q_z_func=z_func)
    negative_field = E_theta(x, z - 2, t + T / 2,
                             q_z_func=z_func)
    # print("time:", time() - t_start)
    field = double_sqrt(positive_field - negative_field)

    cmap = plt.cm.gist_rainbow
    # cmap = plt.cm.viridis

    # norm = plt.Normalize(vmin=double_sqrt(-0.027), vmax=double_sqrt(0.016))
    norm = plt.Normalize(vmin=double_sqrt(-0.02), vmax=double_sqrt(0.01))
    image = cmap(norm(field.T))

    img = np.uint8(image * 255)

    surface = gizeh.Surface.from_image(img)
    circle_up = gizeh.circle(r=9, xy=(1920 / 2, 1080 / 2 + (z_func(t) - 2) * 1080/120), fill=(1,0,0))
    circle_down = gizeh.circle(r=9, xy=(1920 / 2, 1080 / 2 + (z_func(t+T/2) + 2) * 1080/120), fill=(0,0.5,0))
    circle_up.draw(surface)
    circle_down.draw(surface)
    surface.write_to_png(f"out8/{str(next(counter)).rjust(3, '0')}-t={t:.2g}.png")
    # img2 = Image.fromarray(surface.get_npimage())
    # img2.show()

    # plt.imsave(f"out6/{str(next(counter)).rjust(3, '0')}-t={t:.2g}.png", image)

    # plt.clf()
    # plt.axis("off")
    # plt.imshow(field.T, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower",
    #            # vmin=double_sqrt(-0.0332992), vmax=double_sqrt(0.0299505),
    #            vmin=double_sqrt(-0.027), vmax=double_sqrt(0.016),
    #            interpolation="quadric", )
    #
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    #
    # plt.plot([0], [z_func(t) - 2], "ro")
    # # print(pos_z(t), field[25,25])
    # plt.plot([0], [z_func(t + T / 2) + 2], "go")
    # plt.savefig(f"out4/{str(next(counter)).rjust(3, '0')}-t={t:.2g}.png", dpi=1)
    print(f"done t = {t}")


def make_video(img_folder, out="video.mp4"):
    imgs = [os.path.join(img_folder, img) for img in os.listdir(img_folder)]

    clip = ImageSequenceClip(imgs * 5, fps=20)
    clip.write_videofile(out)


def draw_retarded_time(t):
    x, z = np.mgrid[-100:100:2000j, -100:100:2000j]
    time = get_retarded_time_secant(x, z, t, 2)

    plt.clf()
    plt.imshow(time.T, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower")
    plt.plot([0], [z_func(t)], "ko")


# plt.figure(figsize=(1080*RATIO, 1080), dpi=1)
# anim = FuncAnimation(plt.gcf(), draw_retarded_time, np.arange(0, 12, 0.1), interval=100)
# plt.show()

# for t in np.linspace(0, 6, 12*6):
#     draw_dipole_field(t)

make_video("out_quiver", "quiver1.mp4")

# draw_dipole_field(0.9)
