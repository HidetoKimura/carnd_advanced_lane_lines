# Advanced Lane Finding Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[camera_org]: ./camera_cal/calibration2.jpg "Distorted"
[camera_corner]: ./output_images/couners_found/calibration2.jpg "Counersfound"
[camera_undist]: ./output_images/undistorted/calibration2.jpg "Undistorted"
[test1_org]: ./test_images/test1.jpg "Distorted"
[test1_undist]: ./output_images/undistorted/test1.jpg "Undistorted"
[color_grad]: ./files/color_and_grad.png "binarized"
[warped]: ./files/warped.png "warped"
[test2_fit_mask]: ./files/test2_fit_mask.png "masked"
[test2_fit]: ./files/test2_fit.png "fit"
[image_result]: ./files/image_result.png "image result"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I used [this jupyter notebook](https://github.com/HidetoKimura/carnd_advanced_lane_lines/blob/master/project_main.ipynb) in this project. Because in this project to do data analysis and visualization is very important.
Here is the file structures.

### File Structures

project_main.ipynb - The jupyter notebook of the project.    
camera_dist_pickle.p - The pickle file of the camera dist/mtx.  
processed_project_video.mp4 - The result video.  
README.md - This file containing details of the project.  
/files/ - Folder for README.md.  
/output_images/couners_found/ - Folder for camera calibration.  
/output_images/undistorted/ - Folder for undistorted images.  
/output_images/lane_lines/ - Folder for masked lane lines.  


#### Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./project_main.ipynb " 

I start by preparing "objp", which will be the (x, y, z) coordinates of the chessboard corners in the world, and which will be the mesh grid like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0). The "nx, ny" value is　equal to the number of chessboard corners.

If 'cv2.findChessboardCorners' is successful, it can get "corners", which are 2d arrays of coordnates of the chessboard corners, and append to "objpoints" and "imagepoint". You can check these correctness by using "cv2.drawChessboardCorners".

~~~~
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
:
# get grayscale image
:
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, add object points, image points
if ret == True:        
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (nx ,ny), corners, ret)
~~~~

You can use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

~~~~
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_dist_pickle.p", "wb" ) )
:
undist_image = cv2.undistort(img, mtx, dist, None, mtx)
~~~~~
- Original Image
![alt text][camera_org]
- Found Check Corners
![alt text][camera_corner]
- Undistortion Image
![alt text][camera_undist]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The following is the result of applying the undistortion transformation to a test image.
- Original Image
![alt text][test1_org]
- Undistortion Image
![alt text][test1_undist]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the 5th code cell of the IPython notebook located in "./project_main.ipynb "     
I used HLS space for color thresholding. At first I was using only S(saturation) channel.     
But I could not prevent false detection of shadow of "test 5.jpg". And I noticed that the L(Lighting) channel does not detect shadows, so I decided to take a logical AND of S chalnel and L channel. By taking the logical OR of Sx, it was possible to detect the line more beautifully.     
The follow is result. Red is L channel, Blue is S channle and Green is Sx gradient channel.

![alt text][color_grad]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform includes a function called `warp()`. The code for this step is contained in the 7th code cell of the IPython notebook located in "./project_main.ipynb ". 

```
def warp(img,tobird=True):
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    new_top_left=np.array([corners[0,0],0])
    new_top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])    
    if tobird:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)    
    return warped, M
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 340, 720      | 
| 589, 457      | 340, 0        |
| 698, 457      | 995, 0        |
| 1145, 720     | 995, 720      |

I drew a red line on the image of `src` and checked whether it becomes a rectangle after perspective transformation.
In addition, I used ROI to prevent false detection of garbage. The following is result.

![alt text][warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the 10th,12th code cell of the IPython notebook located in "./project_main.ipynb".

Input the image of the lane candidate in `get_masked_lane_image`, divide 6 zones and take the mask with a rectangle of `center_point` and `width` in `get_sliding_window`.     
This `width` is +/- 300 pixel and `center_point` is updated by histogram in the window.

~~~~
[10th cell]
def get_sliding_window(img,center_point,width, thresh=10000):
    """
    function: Mask the rectangle and line candidate specified by center, width, 
              and extract the sliding window.
    input: img,center_point,width,thresh
        img: binary 3 channel image
        center_point: center of window
        width: width of window
        [opt]thresh : threshold of histogram
    
    output: masked,center_point
        masked : a masked image of the same size. mask is a window centered at center_point
        center : the mean of all pixels found within the window
    """

def get_masked_lane_image(binary,center_point,width):
    """
    function: Integrate the sliding windows and take out the masked image.
    input: binary,center_point,width
        binary: binary 3 channel image
        center_point: center of window
        width: width of window
    
    output: window_image
        window_image : a masked image of the same size. mask is a window centered at center_point
    """

~~~~

I extracted the following mask image.

![alt text][test2_fit_mask]

Input the masked image of the lane candidate in `Line::update()`, and calculate the polynomial from all x,y points.
These coefficients are calculated in the world space.
~~~~
[12th cell]
def Line::update(self,lane):
        self.ally,self.allx = (lane[:,:,0]>254).nonzero()
 
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 35.0/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        yvals = self.ally*ym_per_pix
        xvals = self.allx*xm_per_pix
        self.current_fit_coeffs = np.polyfit(yvals, xvals, 2)

        # Calculate xvals
        fit_yvals = self.fit_yvals * ym_per_pix       
        self.current_fit_xvals = (self.current_fit_coeffs[0]*fit_yvals**2 \
            + self.current_fit_coeffs[1]*fit_yvals + self.current_fit_coeffs[2]) / xm_per_pix
            
~~~~
![alt text][test2_fit]


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the 12th code cell of the IPython notebook located in "./project_main.ipynb".
The calcuration is summarized in [this tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).  
For a second order polynomial f(y)=A y^2 +B y + C the radius of curvature is given by R = [(1+(2 Ay +B)^2 )^3/2]/|2A|.

Since I already have the coefficient in world space, I can get the position with the lowest y coordinate.    
The center of the image is set to 670.
~~~~
        y_eval = max(fit_yvals)

        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*self.current_fit_coeffs[0]*y_eval + self.current_fit_coeffs[1])**2)**1.5) \
                         /np.absolute(2*self.current_fit_coeffs[0])

        # Calculate the line base position
        line_base_pos = self.current_fit_coeffs[0]*y_eval**2 \
                        + self.current_fit_coeffs[1]*y_eval \
                        + self.current_fit_coeffs[2]
        self.line_pix_pos = line_base_pos / xm_per_pix
        center_pos = 670 * xm_per_pix
        self.line_base_pos = (line_base_pos - center_pos)
~~~~

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the 15th,16th,17th code cell of the IPython notebook located in "./project_main.ipynb".

"process_image()" passes through the pipeline as explained above and gets `fit lines`.
"project_lane_lines()" projects `fit lines` and combines it into the original image.

~~~~
def project_lane_lines(img,left_fitx,right_fitx,yvals):
def process_image(img):
~~~~~

![alt text][image_result]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed_project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


