import numpy as np
from colour import Color
from customutils2.manimutils.make_scene import make_scene, MEDIUM_QUALITY, HIGH_QUALITY
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeIn, FadeOut
from manimlib.animation.growing import GrowFromCenter
from manimlib.animation.transform import ApplyMethod
from manimlib.animation.update import UpdateFromFunc, UpdateFromAlphaFunc
from manimlib.constants import *
from manimlib.mobject.coordinate_systems import Axes
from manimlib.mobject.geometry import Vector, Circle, Square
from manimlib.mobject.svg.tex_mobject import TexMobject
from manimlib.mobject.types.vectorized_mobject import VGroup, VMobject
from manimlib.scene.scene import Scene
from manimlib.utils.bezier import interpolate
from manimlib.utils.rate_functions import linear, smooth


class Polarization(Scene):
    CONFIG = {
        "period": 6,
        "camera_config": {"background_color": Color("white")},
        "axes_scale": 0.9,
        "vector_length": 2,
        "vector_args": {
            "max_stroke_width_to_length_ratio": 10000,
            "max_tip_length_to_length_ratio": 0.95,
        },
        "xy_line_stroke_width": DEFAULT_STROKE_WIDTH / 2,
        "xy_line_color": Color("blue"),
        "path_color": Color("red"),
        "axes_color": BLACK,
        "axes_vector_color": Color("green"),
        "xy_vector_color": Color("blue"),
    }

    def construct(self):
        self.phase = 0

        point_x, point_y = self.comp_point(0)

        self.make_axes()

        self.ellipse = VMobject(color=self.path_color)

        self.phi = TexMobject(r"\Delta \Phi = ").set_color(BLACK).shift(4 * RIGHT + 2 * UP)
        self.phi_0 = TexMobject("0").set_color(BLACK).next_to(self.phi, RIGHT)
        self.phi_pi_2 = TexMobject(r"\frac{\pi}{2}").set_color(BLACK).next_to(self.phi, RIGHT).shift(0.06 * DOWN)
        self.phi_3_pi_4 = TexMobject(r"\frac{3 \pi}{4}").set_color(BLACK).next_to(self.phi, RIGHT)

        self.x_vector = Vector(point_x * RIGHT, color=self.axes_vector_color, **self.vector_args)
        self.y_vector = Vector(point_y * UP, color=self.axes_vector_color, **self.vector_args)

        self.xy_vector = Vector(point_x * RIGHT + point_y * UP, color=self.xy_vector_color, **self.vector_args)

        update_group = VGroup(self.ellipse,
                              self.xy_vector,
                              self.x_vector,
                              self.y_vector,
                              )

        self.wait(1.6)
        self.add(self.ellipse)

        self.play(FadeIn(self.x_vector))
        # self.wait(1)
        self.play(FadeIn(self.y_vector))
        self.wait(4)
        self.play(FadeIn(self.xy_vector))
        self.play(UpdateFromAlphaFunc(update_group, self.do_vectors_and_ellipse_period),
                  run_time=self.period, rate_func=linear)
        self.play(UpdateFromAlphaFunc(update_group, self.do_vectors_only_period,
                                      run_time=self.period, rate_func=linear))
        self.play(AnimationGroup(UpdateFromAlphaFunc(update_group, self.do_vectors_only_period,
                                                     run_time=self.period, rate_func=linear),
                                 AnimationGroup(FadeIn(self.phi), FadeIn(self.phi_0)),
                                 lag_ratio=0.7))
        self.play(UpdateFromAlphaFunc(update_group, self.do_vectors_only_period,
                                      run_time=self.period, rate_func=linear))

        self.x_vector.set_color(BLACK)
        self.play(UpdateFromAlphaFunc(update_group, self.do_90_phase_shift, rate_func=linear),
                  FadeOut(self.ellipse),
                  FadeOut(self.phi_0),
                  FadeIn(self.phi_pi_2),
                  run_time=self.period / 4)

        self.phase = np.pi / 2
        self.ellipse.set_points([])
        self.x_vector.set_color(self.axes_vector_color)

        self.play(UpdateFromAlphaFunc(update_group, self.do_vectors_and_ellipse_period),
                  run_time=self.period, rate_func=linear)
        for _ in range(3):
            self.play(UpdateFromAlphaFunc(update_group, self.do_vectors_only_period),
                      run_time=self.period, rate_func=linear)

        self.x_vector.set_color(BLACK)
        self.play(UpdateFromAlphaFunc(update_group, self.do_45_phase_shift, rate_func=linear),
                  FadeOut(self.ellipse),
                  FadeOut(self.phi_pi_2),
                  FadeIn(self.phi_3_pi_4),
                  run_time=self.period / 8)

        self.phase = 3 * np.pi / 4
        self.ellipse.set_points([])
        self.x_vector.set_color(self.axes_vector_color)

        self.play(UpdateFromAlphaFunc(update_group, self.do_vectors_and_ellipse_period),
                  run_time=self.period, rate_func=linear)

        self.play(AnimationGroup(UpdateFromAlphaFunc(update_group, self.do_vectors_only_period_hide,
                                                     rate_func=linear, run_time=self.period),
                                 AnimationGroup(FadeOut(self.phi_3_pi_4), FadeOut(self.phi)),
                                 lag_ratio=0.7))
        self.play(FadeOut(self.axes), FadeOut(self.axes_labels))
        self.wait()

    def make_axes(self):
        self.axes = Axes(x_min=-self.axes_scale * FRAME_X_RADIUS,
                         x_max=self.axes_scale * FRAME_X_RADIUS,
                         y_min=-self.axes_scale * FRAME_Y_RADIUS,
                         y_max=self.axes_scale * FRAME_Y_RADIUS,
                         x_axis_config={"include_ticks": False},
                         y_axis_config={"include_ticks": False},
                         number_line_config={"color": self.axes_color}
                         )
        self.axes_labels = self.axes.get_axis_labels(x_label_tex="E_x", y_label_tex="E_y").set_color(self.axes_color)

        self.play(AnimationGroup(GrowFromCenter(self.axes),
                                 FadeIn(self.axes_labels),
                                 lag_ratio=0.4))

    def do_vectors_and_ellipse_period(self, _, alpha):
        point_x, point_y = self.comp_point(alpha)
        self.add_point_to_ellipse(point_x, point_y)
        self.set_vector_positions(point_x, point_y)

    def do_vectors_only_period(self, _, alpha):
        point_x, point_y = self.comp_point(alpha)
        self.set_vector_positions(point_x, point_y)

    def do_90_phase_shift(self, _, alpha):
        alpha = interpolate(0, 1 / 4, alpha)
        x_point, y_point = self.comp_point(alpha)
        self.y_vector.put_start_and_end_on(ORIGIN, y_point * UP)
        self.xy_vector.put_start_and_end_on(ORIGIN, y_point * UP + 2 * RIGHT)

    def do_45_phase_shift(self, _, alpha):
        alpha = interpolate(0, 1 / 8, alpha)
        x_point, y_point = self.comp_point(alpha)
        self.y_vector.put_start_and_end_on(ORIGIN, y_point * UP)
        self.xy_vector.put_start_and_end_on(ORIGIN, y_point * UP + 2 * RIGHT)

    def set_vector_positions(self, x, y):
        self.x_vector.put_start_and_end_on(ORIGIN, x * RIGHT)
        self.y_vector.put_start_and_end_on(ORIGIN, y * UP)
        self.xy_vector.put_start_and_end_on(ORIGIN, x * RIGHT + y * UP)

    def add_point_to_ellipse(self, x, y):
        if not len(self.ellipse.get_points()):
            self.ellipse.set_points([UP * y + RIGHT * x])
        else:
            self.ellipse.add_points_as_corners([UP * y + RIGHT * x])

    def do_vectors_only_period_hide(self, _, alpha):
        fade_ratio = 3
        if alpha >= (fade_ratio - 1) / fade_ratio:
            opacity = 1 - smooth(fade_ratio * alpha - (fade_ratio - 1))
            self.x_vector.set_opacity(opacity)
            self.y_vector.set_opacity(opacity)
            self.xy_vector.set_opacity(opacity)
            self.ellipse.set_stroke(opacity=opacity)
        self.do_vectors_only_period(_, alpha)

    def comp_point(self, alpha):
        point_x = self.vector_length * np.cos(2 * np.pi * alpha) * RIGHT + 0.01 * RIGHT
        point_y = self.vector_length * np.cos(2 * np.pi * alpha + self.phase) * UP + 0.01 * UP
        return point_x, point_y


if __name__ == '__main__':
    make_scene(Polarization,
               quality=HIGH_QUALITY,
               frame_rate=24,
               save_as_gif=True,
               )
