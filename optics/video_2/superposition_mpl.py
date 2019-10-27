import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import numpy as np

mpl.use("qt5Agg")
plt.style.use("C:/Users/tomas/AppData/Roaming/Python/Python37/site-packages/custom_utils/science/latex_style.mplstyle")
mpl.rcParams['toolbar'] = 'None'


def cos(x):
    return np.cos(x * 2 * np.pi)


fig, (upper, medium, lower) = plt.subplots(3, figsize=(19.2/3, 10.8/3), dpi=300, sharex=True)

x = np.linspace(0, 6, 300)


def frame(t):
    t = t / 50
    for ax in [upper, medium, lower]:
        ax.clear()
        ax.set_ylim(-2.1, 2.1)

    y1 = cos(x)
    y2 = cos(x - t)

    upper.plot(x, y1)
    medium.plot(x, y2)
    lower.plot(x, y1 + y2)

    fig.subplots_adjust(top=0.86)

    upper.text(2.5, 2.8, f"$\\Phi = {t:.2f} \\pi$")


    fig.savefig(f"binary/{t:.2f}.png")
    print(f"frame {t}")

for i in range(50, 100):
    frame(i)