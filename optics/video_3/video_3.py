import math

from colour import Color
from manimlib.animation.fading import FadeIn
from manimlib.animation.update import UpdateFromFunc
from manimlib.constants import *
from manimlib.mobject.coordinate_systems import Axes
from manimlib.mobject.geometry import Vector, Ellipse, Circle, Line, DashedLine, Dot, Arc
from manimlib.mobject.svg.brace import Brace
from manimlib.mobject.types.vectorized_mobject import VGroup, VMobject
from manimlib.mobject.value_tracker import ValueTracker
from manimlib.scene.scene import Scene
from manimlib.utils.rate_functions import linear


class EllipseToCircleToLine(Scene):
    CONFIG = {
        "width": 6,
        "color": Color("black")
    }

    def construct(self):
        ellipse = Ellipse(width=self.width, height=self.width / 2)
        circle = Ellipse(width=self.width, height=self.width)
        line = Ellipse(width=self.width, height=0)

        for m in [ellipse, circle, line]:
            m.set_color(self.color)

        self.play(FadeIn(ellipse))
        self.wait()
        self.play(ellipse.become, circle)
        self.wait()
        self.play(ellipse.become, line)
        self.wait()


# noinspection PyAttributeOutsideInit
class PolarizationEllipse(Scene):
    CONFIG = {
        "vector_args": {
            "max_stroke_width_to_length_ratio": 10000,
            "max_tip_length_to_length_ratio": 0.95,
        },
        "width": 6,
        "height": 3,
        "azimuth": PI / 6,
        "axes_color": Color("black"),
    }

    def construct(self):
        axes = Axes(x_axis_config={"include_ticks": False},
                    y_axis_config={"include_ticks": False},
                    number_line_config={"color": self.axes_color}).scale(0.9)
        self.add(axes)

        ellipse = Ellipse(width=self.width, height=self.height)
        ellipse_x_axis = DashedLine(self.width / 2 * LEFT, self.width / 2 * RIGHT,
                                    color=BLACK)
        ellipse_y_axis = DashedLine(self.height / 2 * UP, self.height / 2 * DOWN,
                                    color=BLACK)

        elipticity_line = Line(self.width / 2 * LEFT, self.height / 2 * UP, color=BLUE)
        elipticity_arc = Arc(0, math.atan(self.height / self.width)).shift(self.width / 2 * LEFT)

        ellipse_group = VGroup(ellipse, ellipse_x_axis, ellipse_y_axis, elipticity_arc, elipticity_line)
        ellipse_group.rotate(self.azimuth)

        self.add(ellipse_group)

        rightest = self.get_ellipse_rightest_point(self.width / 2, self.height / 2, self.azimuth)
        highest = self.get_ellipse_highest_point(self.width / 2, self.height / 2, self.azimuth)

        vertical_line = Line(rightest * RIGHT, rightest * RIGHT + highest * UP, color=GREEN)
        horizontal_line = Line(highest * UP, highest * UP + rightest * RIGHT, color=GREEN)
        self.bring_to_back(vertical_line, horizontal_line)

        azimuth_arc = Arc(0, self.azimuth)

        self.add(azimuth_arc)

        self.wait()

    def get_ellipse_rightest_point(self, a, b, theta):
        angle = math.atan(b / a * math.tan(theta))
        return a * math.cos(angle) * math.cos(theta) + b * math.sin(angle) * math.sin(theta)

    def get_ellipse_highest_point(self, a, b, theta):
        angle = math.atan(b / (a * math.tan(theta)))
        return a * math.cos(angle) * math.sin(theta) + b * math.sin(angle) * math.cos(theta)


if __name__ == '__main__':
    from customutils2.manimutils.make_scene import make_scene

    make_scene(PolarizationEllipse, color="grey")
