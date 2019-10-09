from manimlib.animation.creation import ShowCreation
from manimlib.mobject.geometry import Circle
from manimlib.mobject.svg.tex_mobject import TexMobject
from manimlib.scene.scene import Scene


class Test(Scene):

    def construct(self):
        circle = Circle()
        text = TexMobject(r"\pi")

        self.play(ShowCreation(circle))
        self.play(circle.become, text)
        self.wait()


if __name__ == '__main__':
    from customutils2.manimutils.make_scene import make_scene

    make_scene(Test)
