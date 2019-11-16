from moviepy.editor import ImageSequenceClip
import os
import re

image_folder = "out2"
video_name = "quiver.webm"
fps = 20


def key(s):

    r = re.findall("[0-9]+", s)
    return int(r[0])


images = [img for img in os.listdir(image_folder)]
images = [os.path.join(image_folder, img) for img in sorted(images, key=key)]

clip = ImageSequenceClip((images*5), fps=fps)
clip.write_videofile(video_name)
