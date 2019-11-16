import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from moviepy.editor import ImageSequenceClip
from scipy.misc import derivative
from scipy.optimize import root_scalar

import dipole
# import pyximport
# pyximport.install()
# from dipole import *

plt.rcParams["animation.ffmpeg_path"] = "ffmpeg"

C = 2
EPS_0 = 1
Q = 1
T = 6


def get_retarded_time(x, z, q_x_func, q_z_func, t):
    def func(t_r):
        return np.sqrt((x - q_x_func(t_r)) ** 2 + (z - q_z_func(t_r)) ** 2) - C * (t - t_r)

    result = root_scalar(func, x0=0, x1=1)

    return result.root


def pos_x(t):
    return 0 * t


def pos_z(t):
    return np.sin(2 * np.pi * t / T)


def v_x(t):
    return derivative(pos_x, t, dx=1e-6)


def v_z(t):
    return derivative(pos_z, t, dx=1e-6)


def a_x(t):
    return derivative(pos_x, t, dx=1e-6, n=2)


def a_z(t):
    return derivative(pos_z, t, dx=1e-6, n=2)


def E(x, z, t):
    t_ret = get_retarded_time(x, z, pos_x, pos_z, t)
    nice_r_x = x - pos_x(t_ret)
    nice_r_z = z - pos_z(t_ret)

    nice_r_norm = np.sqrt(nice_r_x ** 2 + nice_r_z ** 2)

    v_x_ret = v_x(t_ret)
    v_z_ret = v_z(t_ret)
    a_x_ret = a_x(t_ret)
    a_z_ret = a_z(t_ret)

    u_x = C * nice_r_x / nice_r_norm - v_x_ret
    u_z = C * nice_r_z / nice_r_norm - v_z_ret

    nice_r_dot_u = nice_r_x * u_x + nice_r_z * u_z
    const = Q / (4 * np.pi * EPS_0)
    front = nice_r_norm / (nice_r_dot_u ** 3)

    c_squared_minus_v_squared = C ** 2 - (v_x_ret ** 2 + v_z_ret ** 2)

    # coulomb_term_x = c_squared_minus_v_squared * u_x
    # coulomb_term_z = c_squared_minus_v_squared * u_z

    radiation_term_x = -nice_r_z * (u_z * a_x_ret - u_x * a_z_ret)
    radiation_term_z = nice_r_x * (u_z * a_x_ret - u_x * a_z_ret)

    E_x = const * front * (radiation_term_x)
    E_z = const * front * (radiation_term_z)

    theta = np.arctan2(abs(x), z)
    e_theta_x = np.cos(theta)
    e_theta_z = -np.sin(theta)

    # return np.sqrt(E_x**2 + E_z**2)
    # return E_x, E_z
    return E_x * e_theta_x * np.sign(x) + E_z * e_theta_z


def draw_retarded_time(t):
    x, z = np.mgrid[-3:3:50j, -3:3:50j]
    time = np.vectorize(get_retarded_time)(x, z, pos_x, pos_z, t)

    plt.clf()
    plt.imshow(time.T, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower")
    plt.plot([pos_x(t)], [pos_z(t)], "ko")


def draw_E(t):
    global counter

    halfside = 25
    x = np.linspace(-halfside, halfside, 30)
    z = np.linspace(-halfside, halfside, 30)
    xm, zm = np.meshgrid(x, z)

    # field_x = xm * 0
    # field_z = zm * 0
    # for i, _ in enumerate(field_x):
    #     for j, _ in enumerate(field_x[0]):
    #         if abs(i - 14.5) < 5.5 and abs(j - 14.5) < 2.5:
    #             field_x[i, j] = np.nan
    #             field_z[i, j] = np.nan
    #             continue
    #         e1 = E(x[j], z[i] + 2, t)
    #         e2 = E(x[j], z[i] - 2, t + T / 2)
    #         field_x[i, j] = e1[0] - e2[0]
    #         field_z[i, j] = e1[1] - e2[1]

    # field1_x, field1_z = dipole.E_components_on_grid(
    #     x, z, t, pos_x, pos_z,
    #     z_origin=2,
    #     mask_func=lambda x_, z_: abs(x_) < 1 and abs(z_) < 4
    # )
    # field2_x, field2_z = dipole.E_components_on_grid(
    #     x, z, t, pos_x, pos_z,
    #     z_origin=-2, t_origin=T/2,
    #     mask_func=lambda x_, z_: abs(x_) < 1 and abs(z_) < 4
    # )
    # field_x = field1_x - field2_x
    # field_z = field1_z - field2_z
    field_x, field_z = dipole.E_components_on_grid(
        x, z, t, pos_x, pos_z,
        mask_func=lambda x_, z_: abs(x_) < 1 and abs(z_) < 4
    )

    plt.clf()
    # plt.imshow(field.T, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower",)
    # vmin=0, vmax=1)
    # cs = plt.contourf(xm, zm, field_z,
    #                   # colors="k",
    #                   extend="both",
    #                   levels=[-0.02, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.02])
    plt.quiver(xm, zm, field_x, field_z)
    # plt.clabel(cs)
    plt.plot([pos_x(t)], [pos_z(t)], "ro")
    # plt.plot([pos_x(t + T / 2)], [pos_z(t + T / 2) + 2], "bo")
    plt.tight_layout()
    # plt.savefig(f"out2/{counter}.png")
    counter += 1
    print(f"done t = {t}")


def draw_E_dipole(t):
    halfside = 15
    x, z = np.mgrid[-halfside:halfside:50j, -halfside:halfside:50j]
    E_vect = np.vectorize(E)
    field1 = E_vect(x, z, t)
    field2 = E_vect(x, z, t + T / 2)
    field = field1 - field2

    plt.clf()
    # plt.imshow(field.T, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower",)
    # vmin=0, vmax=1)
    cs = plt.contourf(x, z, field,
                      # colors="k",
                      extend="both",
                      levels=[-0.02, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.02])
    # plt.clabel(cs)
    plt.plot([pos_x(t)], [pos_z(t)], "ro")
    plt.plot([pos_x(t + T / 2)], [pos_z(t + T / 2)], "go")


def double_sqrt(x):
    return np.sqrt(abs(x)) * np.sign(x)

counter = 0

def draw_E_dipole2(t):
    global counter

    halfside = 60
    x, z = np.mgrid[-halfside:halfside:50j, -halfside:halfside:50j]
    E_vect = np.vectorize(E)
    field1 = E_vect(x, z + 2, t)
    field2 = E_vect(x, z - 2, t + T / 2)
    # field = field1 - field2
    # x = np.linspace(-halfside, halfside, 50)
    # z = np.linspace(-halfside, halfside, 50)
    # t1 = time()
    # field1 = dipole.E_theta_on_grid(x, z, t, pos_x, pos_z, z_origin=-2)
    # field2 = dipole.E_theta_on_grid(x, z, t, pos_x, pos_z, z_origin=2,
    #                                 t_origin=-T/2)
    # print(time() - t1)
    field = double_sqrt(field1 - field2)

    plt.clf()
    plt.axis("off")
    threshold = 0.2
    plt.imshow(field, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower",
               # vmin=double_sqrt(-0.0332992), vmax=double_sqrt(0.0299505),
               vmin=double_sqrt(-0.03), vmax=double_sqrt(0.025),
               interpolation="quadric",)
               # cmap="inferno")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    plt.plot([pos_x(t)], [pos_z(t) - 2], "ro")
    # print(pos_z(t), field[25,25])
    plt.plot([pos_x(t + T / 2)], [pos_z(t + T / 2) + 2], "go")
    plt.savefig(f"out3/{str(counter).rjust(3, '0')}-t={t:.2g}.png", bbox_inches="tight")
    print(f"done t = {t}")
    counter += 1


def make_video(img_folder, out="video.mp4"):
    imgs = [os.path.join(img_folder, img) for img in os.listdir(img_folder)]

    clip = ImageSequenceClip(imgs * 5, fps=10)
    clip.write_videofile(out)

plt.figure(figsize=(6, 6), dpi=100)
anim = FuncAnimation(plt.gcf(), draw_E_dipole2, np.arange(0, 12, 0.1), interval=100)
# anim.save("anim_E_theta_large.mp4")
plt.show()
