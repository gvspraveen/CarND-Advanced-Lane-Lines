## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[images_with_corners]: ./submission/images/images_with_corners.jpg "Images with Corners"
[distortion_correction]: ./submission/images/distortion_correction.jpg "Distortion Correction"
[warp_unwarp]: ./submission/images/warp_unwarp.jpg "Warp and Unwarp"
[mag_dir_threshold]: ./submission/images/mag_dir_threshold.jpg "Mag and Direction"
[l_s_grad_threshold]: ./submission/images/l_s_grad_threshold.jpg "L + S + Mag and Direction"
[y_w_threshold]: ./submission/images/y_w_threshold.jpg "Y + W Threshold"
[color_ls_grad]: ./submission/images/color_ls_grad.jpg "color_ls_grad"
[line_fit_no_current]: ./submission/images/line_fit_no_current.jpg "Y + W Threshold"
[line_fit_with_current_fit]: ./submission/images/line_fit_with_current_fit.jpg "line_fit_with_current_fit"
[final_lines_drawn]: ./submission/images/final_lines_drawn.jpg "final_lines_drawn"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook - **solution.ipynb**.

I divided notebook into multiple sections by adding headers. Camera calibration implementation is located in section - **Step 1: Calibrate camera**

I first explored the images in `1.1 Explore Calbiraton images`.

Calibration process is very similar to what is taught in the classroom videos. Major difference here is we have 9x6 chessboard.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. 
 Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  
 `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I used the function
 `cv2.drawChessboardCorners` to draw corners on the chessboard.

The logic for all this is in method `get_camera_corners` under section - **1.2 Calibrate camera**. This function returns a tuple of 
`(img_with_corners, objpoints, imgpoints)`. Here is a out put of images with corners drawn.

![alt text][images_with_corners]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The logic for this part is in section - **Step 2: Define a function to undistort images** in notebook.

I use output from `get_camera_corners` (described above) to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

I then defined a func - `undistort_image` which uses coefficients from before and returns a undistored image. It uses `cv2.undistort` to achieve this.

Here is some sample output from distortion correction.

![alt text][distortion_correction]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform lives in section - **Step 3: Warp and unwarp images**.

##### Explore test images.
I first explored various test images to get a rough idea of where boundaries of polygon should be. You can see this
in section - **3.1 Explore test images**

##### Define Perspective Transform function 

In section - **3.2 Define Perspective Transform function**, defined two functions `warp_image` and `do_perspective_transform(image, src, dst)`.
`warp_image` just uses `cv2.warpPerspective` to warp image. You could provide M or M_inv to warp or unwarp.

`do_perspective_transform(image, src, dst)` computes `M` and `M_inv`. It calls `warp_image` and returns a tuple of `(warped_image, m_inv)`. 

##### Explore various warp points 

I then use these two functions and explored various warp points and outputted result image.
On playing around I found following src and dst. I noted that total image size is `720, 1280, 3`.
 
```python
height, width, _ = img.shape
# 720, 1280, _
src = np.float32([(600.,450.),
                      (720.,450.), 
                      (255.,680.),
                      (1100.,680.) 
                      ])

dst = np.float32([(445.,0),
                  (width-445.,0),
                  (445.,height),
                  (width-445.,height)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600., 450.    | 445., 0       | 
| 720., 450.    | 835., 0.      |
| 255., 680.    | 445., 720.    |
| 1100., 680.   | 835., 720.    |

Here is some output images using this src and dst.

![alt text][warp_unwarp]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this section lives in section - **Step 4: Binary Thresholds for the Image` in python notebook**

Note: I defined bunch of helper functions for thesholds in `binary_thresholds.py`. In this python file I defined,
individual threshold functions (like sobelx, mag, thresh, l etc). I then used them to try out combination
function in python notebook.

Finally the key observations which led to my solution are

##### Using mag + direction threshold

Using a combination of mag & direction threshold, seemed to give a fairly decent detection. But when
lighting is poor (shadows etc) it did not do a great job.

Here is sample of output from function `sobel_mag_dir_threshold` in python notebook.


![alt text][mag_dir_threshold]


##### Combining mag+direction with l + s (from HLS)

To solve the problem where mag+threshold does not do well in shadows and light color on lines, I combined
with L and S threshold. This ensured that light green is captured well in shady conditions.
But this still did not great job with white lines which were super light in color. 

Code can be seen in method `l_s_grad_threshold` method in jupyter notebook. Here are some sample
images from this threshold. You can see when compared to pure mag_dir_threshold this does better job of
capturing yellow lines in shady regions.

However, you can see from images white lines in shady regions are still a problem.

![alt text][l_s_grad_threshold]

##### Using Yellow and White detectors

I read some articles on using Laplacian transform etc for best detection. There is even a open cv
function for it apparantly.

But I wanted to use something I understand. So on reading more, I got idea of detecting yellow and
white lines in image. So in order to solve the issue with white lines from previous section, I wrote a
threshold function to purely detect yellow and white lines. I got this idea from [finding-lane-lines-with-colour-thresholds](https://medium.com/@tjosh.owoyemi/finding-lane-lines-with-colour-thresholds-beb542e0d839). 

You can see the code for this in `y_w_threshold` method in jupyter notebook. Here is output from this.

You will notice that this does better job with white lines but yellow lines in shade get cut out in corners.

![alt text][y_w_threshold]


##### Combined Threshold

My final threshold function involves combining previous functions (l_s_grad_threshold, y_w_threshold) to cover all use cases.

Code be found in method `combined_threshold`. This gave me best results.

![alt_text][color_ls_grad]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this under section - **Step 5: Finding Lanes**

The code is basically divided into two helper functions

```
1. sliding_window_search(binary_image, draw_boxes=False)
2. polyfit_lanes(binary_image, 
                      current_left_fit=None, 
                      current_right_fit=None,
                      draw_boxes=False
                     ) 

```

`polyfit_lanes` uses `sliding_window_search` internally. In case previous fits are present, then we dont
need to search for windows of lane. 
 
`sliding_window_search` is very similar to what has been taught in the class. Comments in the method
explain the detail. We basically use histogram from binary threshold iamge to detect the starting position
for left and right locations. We then use a windows from bottom of image to look for non zero pixels
in left and right lanes. We capture all these lane pixel indices.
Later using these indices we polyfit lane line.

I ran this on a two subsequent frames for verification. For first one there is no previous fit. And second one uses
fit from previous frame. Here are the results.


![alt text][line_fit_no_current]

![alt text][line_fit_with_current_fit]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This part is under section **5.2: Determine radius of curvature and center offset** in jupyter notebook.

I first implemented a method to provide meter per pixel in x and y direction. By observing the images
from previous steps, I could figured lane size is about 380px in width and 720 px and height.

From U.S. regulations width of lane is about 3.7 meters, and based on instructions vertical lenght is about
30m. Using these, I could calculate meters per pixel. You can see method `get_meters_per_pixel`.

I then defined two methods

1. get_curvature: Which defines curvature of road
2. get_center_offset: which figures out position of car from middle of lane.

**get_curvature** is pretty similar to instructions in lecture.
**get_center_offset** is just `image x midpoint - mean of left_fit and right_fit intercepts`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This part is in section **5.3 Define Pipeline class which processes each image**

Specifically in method `draw_detected_lane(self, frame_img, binary_img, m_inv)` which is part of Pipeline class.

The logic for this is as follows.

1. Get the current best left and right fit
2. Get a copy of warped binary image and create a colored warp (3 dimensonal space for rgb). Figure out left and righ lane pixel points from fits.
3. Use `cv2.fillPoly` to draw poly lines on warped color image.
4. Overlay this on top of copy of original frame's color image. Used `cv2.addWeighted`
5. Finally get `mean_curvature` and `car_offset` and write a text on top left corner of image.

Here is output

![alt text][final_lines_drawn]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Output video is uploaded to [project_video_out.mp4](./project_video_out.mp4).

You can watch it in-line at [project_video_out.html](./project_video_out.html).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent a lot of time trying to figure out thresholds for the image. The big challenge for me was,
video has about 1260 frames of different lighting conditions. Sometimes there were shadows and sometimes
lanes were not marked properly.

I had to play around with a lot of those images and different combination. I have explained final approach and resources
I read in earlier section in the read me.

After this other challenge was to make sure lines are fairly smooth from frame to frame. I used techniques to 
average out previous n fits and consiter that as best fit on every frame to provide smoother lines. Apart from
that I needed to guard against cases where I dont find left or right fit etc.

After all this code worked fairly well on project video. But it still did not perform great on challenge and harder challenge.
Reason was, I needed more dynamic thresholds. Basically, I need to define different combinations of thresholds. Dynamically
at run time consider each one and figure out the error of fit. By comparing fit from previous lane
and making sure we are not off by lot. At each frame pick a threshold combination of thresholds which minimizes
error w.r.t to previous frame's fit.

May be I will try that sometime in future. But for now, I got a good result on current project video. I already spent lot of time on this to get this far.

Overall, I wish there was a better way to do this. Current approach seemed very manual and lot of tuning which after a while
got little painful. 


