from manimlib.mobject.value_tracker import ValueTracker
from manimlib.scene.scene import Scene


class PolarizationEllipse(Scene):

    def construct(self):
        theta = ValueTracker(0)



if __name__ == '__main__':
    from customutils2.manimutils.make_scene import make_scene

    make_scene(PolarizationEllipse, color="white")
