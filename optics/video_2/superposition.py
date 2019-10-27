import numpy as np
from colour import Color
from customutils2.manimutils.make_scene import make_scene, HIGH_QUALITY, MEDIUM_QUALITY
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.creation import ShowCreation
from manimlib.animation.fading import FadeIn
from manimlib.animation.transform import Transform
from manimlib.animation.update import UpdateFromAlphaFunc
from manimlib.mobject.coordinate_systems import Axes, NumberPlane
from manimlib.mobject.functions import FunctionGraph
from manimlib.mobject.geometry import Line, DashedLine
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.scene.scene import Scene
from manimlib.constants import *
from manimlib.utils.bezier import interpolate
from manimlib.utils.rate_functions import linear, double_smooth, there_and_back, running_start, not_quite_there, wiggle, \
    lingering, exponential_decay, smooth, rush_into, rush_from

UPPER_POSITION = 4.5 * LEFT + 2.4 * UP
MIDDLE_POSITION = 4.5 * LEFT + 0.45 * UP
LOWER_POSITION = 4.5 * LEFT + 2 * DOWN

UPPER_COLOR = Color("red")
MIDDLE_COLOR = Color("blue")
LOWER_COLOR = Color("purple")


class Superposition(Scene):

    def construct(self):
        self.phase = 0

        big_wave = self.make_graph(lambda x: self.upper_func(x) * 3, UPPER_COLOR).shift(4.5 * LEFT)
        self.play(ShowCreation(big_wave), run_time=2, rate_func=lambda t: running_start(t, pull_factor=1))
        self.wait()

        upper_axes = self.make_axes(0.9, "E_1").shift(UPPER_POSITION)
        self.upper_graph = self.make_graph(self.upper_func, UPPER_COLOR).shift(UPPER_POSITION)

        middle_axes = self.make_axes(0.9, "E_2").shift(MIDDLE_POSITION)
        self.middle_graph = self.make_graph(self.middle_func, MIDDLE_COLOR).shift(MIDDLE_POSITION)

        lower_axes = self.make_axes(1.4, "E_1 + E_2").shift(LOWER_POSITION)
        self.lower_graph = self.make_graph(self.lower_func, LOWER_COLOR).shift(LOWER_POSITION)

        self.play(AnimationGroup(Transform(big_wave, self.upper_graph, run_time=1.2),
                                 FadeIn(upper_axes), lag_ratio=0.5))
        self.bring_to_back(upper_axes)

        self.play(FadeIn(middle_axes), FadeIn(self.middle_graph))
        # self.wait()
        self.play(FadeIn(lower_axes), FadeIn(self.lower_graph))
        self.wait(0.5)

        self.play(UpdateFromAlphaFunc(VGroup(self.middle_graph, self.lower_graph), self.anim_pi_2),
                  run_time=2, )
        self.wait(0.5)
        self.play(UpdateFromAlphaFunc(VGroup(self.middle_graph, self.lower_graph), self.anim_pi_3),
                  run_time=1.5, )
        self.wait()
        self.wait()

    def make_axes(self, y, label):
        axes = Axes(x_min=0,
                    x_max=9.5,
                    y_min=-y,
                    y_max=y,
                    number_line_config={
                        "color": Color("black"),
                        "include_ticks": False
                    })
        label = axes.get_y_axis_label(label_tex=label, direction=DL).set_color(Color("black"))
        return VGroup(axes, label)

    def make_graph(self, func, color):
        return FunctionGraph(func, x_min=0, x_max=9, color=color)

    def upper_func(self, x):
        return np.cos(x * 2) / 2

    def middle_func(self, x):
        return self.upper_func(x - self.phase)

    def lower_func(self, x):
        return self.upper_func(x) + self.middle_func(x)

    def anim_pi_2(self, _, alpha):
        self.anim_func(_, alpha, 0, np.pi / 2)

    def anim_pi_3(self, _, alpha):
        self.anim_func(_, alpha, np.pi / 2, np.pi / 3)

    def anim_pi_something(self, _, alpha):
        self.anim_func(_, alpha, np.pi / 3, 10 * np.pi / 13)

    def anim_func(self, _, alpha, from_phase, to_phase):
        self.phase = interpolate(from_phase, to_phase, alpha)
        new_middle = self.make_graph(self.middle_func, MIDDLE_COLOR).shift(MIDDLE_POSITION)
        new_lower = self.make_graph(self.lower_func, LOWER_COLOR).shift(LOWER_POSITION)
        self.middle_graph.become(new_middle)
        self.lower_graph.become(new_lower)


if __name__ == '__main__':
    make_scene(Superposition,
               color="white",
               quality=MEDIUM_QUALITY,
               frame_rate=24,
               # stop_at_animation_number=5,
               )
