**Vehicle Detection Project**

The goals / steps of this project are the following:

1. Feature extraction using: spatial binning, color histogram, and HOGs
2. Normalize the features and randomize the data into training and testing sets
3. The train a linear SVM classifier
4. Implement a sliding window technique to search for vehicles in the image
5. Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject false positives and follow detected vehicles.
6. Estimate a bounding box for vehicles detected.
7. Keep track of cars to limit the search area in subsequent frames to speed-up the algorithm (by limiting search area) using vehicle and frame_tracking classes

# (Image References)
./output_images/car_notcar.png
./output_images/HOG_car_examples.png
./output_images/HOG_noncar_examples.png
./output_images/slidingwindow.png
./output_images/out_test1.png
./output_images/out_test2.png
./output_images/out_test3.png
./output_images/out_test4.png
./output_images/out_test5.png
./output_images/out_test6.png
./output_project_video_0212b.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section labelled "Extract Features (HOG, Color, and Spatial Binning)."

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

./examples/car_notcar.png

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

./examples/HOG_car_examples.png
./examples/HOG_noncar_examples.png


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and color spaces to find the best performing features. It was partly alot of trial and error and partly using the intuition developed from reading the Dalal and Triggs' HOG paper as well as the color space intuition I developed from the previous project.

I finally decided to use YCrCb color space and use all 3 channels for HOG, using spatial binning (32x32) and color histograms of all 3 channels -- as it gave the best perfromance.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I normalized the dataset using SKlearn's standard scaler and thereafter I trained a linear SVM using using sklearn's linear svm classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Given there are no cars on the image horizon, I am searching for cars in the bottom half of the image. I chose two window sizes (96,96) and (48,48) as they are closer in size to the cars being detected. Here's an example image:

./examples/slidingwindow.png

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I properly thresholded the heatmap to get rid of any false positives. Here are some example images:

./output_images/out_test1.png
./output_images/out_test2.png
./output_images/out_test3.png
./output_images/out_test4.png
./output_images/out_test5.png
./output_images/out_test6.png

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result (./output_project_video_0212b.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Vehicle Detection: I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurement's label function to idenitify individual groups of regions (blobs) in the heatmap. I then created another heatmap over multiple frames (10 frames) to get a stronger heat map and then thresholded it. This allowed me to reduce the false positives and also give a smoother bounding box. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Vehicle Tracking: To improve the speed of the algorithm, after detecting a few frames with high confidence (i.e. having a good history of seeing all the cars in the current frame), I track the position of all the detected vehicles in the current frame and use it to limit the search window in the next frame. I alternate between doing fast search (tracking based) and full search every few frames (like 4 frames of fast search for every one frame of full search). This works because we have 25 frames per second and cars are not moving too fast to really change much in 100s of milliseconds. I implemented this by creating two classes: vehicle class and frame_tracking class.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The one problem is that this pipeline doesn't work in real time. Given more time, one way I can improve is by doing HOG extraction on the whole image first and then slide the window -- as running HOG is the main bottleneck currently. Another thing I could try is use a neural nets based approach such as a YOLO architecture that doesn't need to slide through the image so many times, and hence will be faster.

