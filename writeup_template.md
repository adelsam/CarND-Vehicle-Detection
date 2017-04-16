## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/image0000.png
[image2]: ./examples/extra40.png
[image3]: ./examples/sub-sample.png
[image4]: ./examples/frame.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I trained the classifier by extracting HOG features in `train_classifier.py`.  The extract_features function does this.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of a `vehicle` image:

![alt text][image1]

And here is an example of a `non-vehicle` image:

![alt text][image2]

I concatenated the vector of spatial binning, color histogram, and hog features to generate the features for classification.

#### 2. Explain how you settled on your final choice of HOG parameters.

I mostly used the parameters I discovered in the Search and Classify lesson, and various others.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the Linear SVM in `train_classifier`.  I decided to use the HOG features in addition to spatial features, and color histogram features.  This gave the following results:
```
Car Images 8792.  Extra Images 8968.
42.23 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 2580
8.4 Seconds to train SVC...
Test Accuracy of SVC =  0.9676
My SVC predicts:  [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
For these 10 labels:  [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
0.00161 Seconds to predict 10 labels with SVC
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `find_cars` method of `car_finder.py`.  I used subsampling as shown in the lessons.  The results for an individual frame look like this:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the following frame (from the test_video clip), the top-left hand image shows the boxes classified as a `vehicle` by the SVM.  The middle top image shows the heatmap create from those boxes, and the top-right shows the output labeled using `scipy.ndimage.measurements.label()` from a thresholded heatmap.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter the video, I used a queue (`video_pipeline.py`, line 53) to store the bounding boxes detected in 8 consecutive frames.  I then combined the bounding boxes from all of these frames into a single heatmap (line 71-72), and used a threshold of 6 overlapping boxes to filter for false positives.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline has trouble detecting the white car when going over the bridge (0:22).  It also has trouble detecting both cars when the black car enters the frame (0:28).  I think I could improve the pipeline by using a different set of color channels and adding some telemetry data (i.e. relative speed of other vehicles detected).

