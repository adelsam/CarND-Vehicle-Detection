import cv2
import glob
import pickle
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from source.car_finder import Finder

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

FRAMES = 10


class Pipeline():
    def __init__(self, input_video, output_video, debug=False):
        self.input_video = input_video
        self.output_video = output_video
        self.debug = debug
        self.car_finder = Finder()
        self.car_bboxes = collections.deque(maxlen=FRAMES)

    def process_image(self, image):
        scale = 1.5
        hog_subsample, box_list = self.car_finder.find_cars(image, scale)

        draw_img, heatmap = self.calc_heatmap(box_list, image)

        if self.debug:
            return self.debug_frame(draw_img, [hog_subsample, heatmap])
        return draw_img

    def calc_heatmap(self, box_list, image):
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list

        #Averaging:
        self.car_bboxes.append(box_list)
        for frame_box_list in self.car_bboxes:
            heat = add_heat(heat, frame_box_list)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, FRAMES - 4)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        return draw_img, heatmap

    def debug_frame(self, overlay, tiles):
        vis = np.zeros_like(overlay)
        if len(tiles) > 0:
            tile_small = cv2.resize(tiles[0], (426, 240))
            vis[:240, :426] = tile_small
        if len(tiles) > 1:
            tile_small = cv2.resize(tiles[1], (426, 240))
            if len(tile_small.shape) == 2:
                tile_small = np.dstack((tile_small, tile_small, tile_small)) * 255
            vis[:240, 426:852] = tile_small
        if len(tiles) > 2:
            tile_small = cv2.resize(tiles[2], (426, 240))
            vis[:240, 852:1278] = tile_small

        # vis [h, w]
        overlay_resized = cv2.resize(overlay, (853, 480))
        vis[240:, :853] = overlay_resized  # bottom-left

        return vis

    def process(self):
        print('Reading Video file {}'.format(self.input_video))
        clip1 = VideoFileClip(self.input_video)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(self.output_video, audio=False)
        print('Wrote Output Video file {}'.format(self.output_video))

if __name__ == '__main__':
    pipeline = Pipeline('../project_video.mp4', '../output_project_video.mp4', debug=False)
    pipeline.process()
