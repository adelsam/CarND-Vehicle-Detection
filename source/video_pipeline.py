import cv2
import glob
import pickle
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from source.car_finder import Finder


class Pipeline():
    def __init__(self, input_video, output_video, debug=False):
        self.input_video = input_video
        self.output_video = output_video
        self.debug = debug
        self.car_finder = Finder()
        # self.left_results = collections.deque(maxlen=5)
        # self.right_results = collections.deque(maxlen=5)

    def process_image(self, image):
        scale = 1.5
        return self.car_finder.find_cars(image, scale)

    def process(self):
        print('Reading Video file {}'.format(self.input_video))
        clip1 = VideoFileClip(self.input_video)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(self.output_video, audio=False)
        print('Wrote Output Video file {}'.format(self.output_video))

if __name__ == '__main__':
    pipeline = Pipeline('../test_video.mp4', '../output_test_video.mp4')
    pipeline.process()
