import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from source.train_classifier import bin_spatial, color_hist, get_hog_features, get_img_features


class Finder():
    def __init__(self):
        with open('svc_fit.p', 'rb') as f:
            dist_pickle = pickle.load(f)
            self.svc = dist_pickle["svc"]
            self.X_scaler = dist_pickle["scaler"]
            self.orient = dist_pickle["orient"]
            self.pix_per_cell = dist_pickle["pix_per_cell"]
            self.cell_per_block = dist_pickle["cell_per_block"]
            self.spatial_size = dist_pickle["spatial_size"]
            self.hist_bins = dist_pickle["hist_bins"]
            self.color_space = dist_pickle["color_space"]
            self.bins_range = dist_pickle["bins_range"]
            self.spatial_feat = dist_pickle["spatial_feat"]
            self.hist_feat = dist_pickle["hist_feat"]
            self.hog_feat = dist_pickle["hog_feat"]
            self.y_start_stop = dist_pickle["y_start_stop"]
            self.hog_channel = dist_pickle["hog_channel"]

    def convert_color(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, scale):

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        ystart = self.y_start_stop[0]
        ystop = self.y_start_stop[1]

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1
        nfeat_per_block = self.orient * self.cell_per_block ** 2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                file_features = get_img_features(subimg, self.color_space, self.spatial_size, self.hist_bins,
                                 self.bins_range, self.hog_channel,
                                 self.orient, self.pix_per_cell, self.cell_per_block,
                                 spatial_feat=True, hist_feat=True, hog_feat=False)
                file_features.append(hog_features)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(
                    np.hstack((file_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

        return draw_img

    def find(self):
        # Find cars in image using sliding-window technique
        scale = 1.5
        img = mpimg.imread('../test_images/test6.jpg')
        out_img = self.find_cars(img, scale)

        plt.imshow(out_img)
        plt.show()

if __name__ == '__main__':
    finder = Finder()
    finder.find()