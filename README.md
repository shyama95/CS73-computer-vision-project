# 3D Character Animation from a Photo
## Abstract
The goal of the project is to generate a 3D animation for the human figure from a single input image. The project should be able to detect, 3D model and animate the human figure from a single input image as long as a frontal view of the person is given in the image. The project will be an implementation of [1].
The steps involved can be briefed as follows :
- Person detection
- Person segmentation
- 2D skeleton estimation
- Fitting morphable (semi nude) posable 3D model
- Generation of rigged body mesh
- Inpainting background
- Motion transferring
## Work Status
We have individually completed the the following sections of the project:
- Person detection and Person segmentation
- Keypoint detection (pose estimation)
- Generation of base 3D model  
### To do
- Texture mapping to initial 3D model
- Model animation
## Dependencies
### Person detection and segmentation code
- python v3
- jupyter-notebook
- python libraries : opencv, numpy, urllib, tarfile, tensorflow, zipfile, matplotlib, pillow
### Keypoint detection code
- python v3
- python libraries : opencv v4.0.0 (contrib)
### Code for generating base 3D model
- python v2.7
- python libraries : opencv v3.4.2, pickle, numpy, chumpy, opendr
## Instructions to run
### Person detection and segmentation code
Run the jupyter-notebook file **src/person segmentation/MaskRCNN.ipynb**. It takes **images/input_image1.jpeg** as input and generates the segmented output at **src/person segmentation/segmented_out.jpg**.
### Keypoint detection code
Download the model from <a href="https://drive.google.com/open?id=1yEeX7NiJ3BkONWe2MIO4UUJRFHJ_s_4S">here</a> and copy it to folder **src/keypoint estimation/mpi**.  
Navigate to **src/keypoint estimation** folder and run the following command:  
`python3 keypoint_estimation.py`  
This file takes the image **images/input_image1.jpeg** as input and generates the output image **src/keypoint estimation/output_keypoints.jpg** with keypoints marked. It also generates the text file **src/keypoint estimation/out_keypoints.txt** with keypoint coordinates and confidence levels, reordered for using in generation of 3D model.
### Code for generating base 3D model
The code for generation is taken from [5].  
Steps:  
- Update the keypoint coordinates and confidence levels from **src/keypoint estimation/out_keypoints.txt** in the code.  
- Navigate to **src/3d model generation** folder and run the following command:  
`python demo.py`  
This file takes the image **images/input_image1.jpeg** as input and generates the initial 3D model at **src/3d model generation/3d_model.png** with keypoints marked.
## Results
The results obtained are given below.
### Person detection and segmentation
Person segmentation is done using a pre-trained Mask RCNN model from tensorflow trained on COCO dataset [6]. Sample result is given below:  
<img src="https://github.com/shyama95/CS73-computer-vision-project/blob/master/images/input_image1.jpeg" width="250" alt="Input image"/> <img src="https://github.com/shyama95/CS73-computer-vision-project/blob/master/images/rcnn_output_image1.jpg" width="250" alt="Segmentation output"/>
### Keypoint detection
Keypoint detection is done using the openpose implementation from opencv using pre-trained weights on MPII Human Pose Dataset [2]. The library provides 15 keypoint locations and their confidence values. This data was given as input to the SMPL library to obtain the intial 3D model. Sample result is given below.  
<img src="https://github.com/shyama95/CS73-computer-vision-project/blob/master/images/input_image1.jpeg" width="250" alt="Input image"/> <img src="https://github.com/shyama95/CS73-computer-vision-project/blob/master/images/keypoint_detection_image1.png" width="250" alt="Keypoint detection output"/>
### Generating base 3D model
SMPL library was used to generate the initial 3D model. It takes the keypoint locations and confidence values as input. The library uses 14 out of 15 keypoints provided by the opencv openpose implementation. These keypoints were rearranged as required by the library. Sample result is given below.  
<img src="https://github.com/shyama95/CS73-computer-vision-project/blob/master/images/input_image1.jpeg" width="250" alt="Input image"/> <img src="https://github.com/shyama95/CS73-computer-vision-project/blob/master/images/smpl_output_image1.png" width="250" alt="Initial 3D model"/>
## References
[1] Weng, Chung-Yi, Brian Curless, and Ira Kemelmacher-Shlizerman. ”Photo wake-up: 3d character animation from a single photo.” arXiv preprint arXiv:1812.02246 (2018).  
[2] Pose Estimation (keypoint detection) : https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/  
[3] SMPL Library : Loper, Matthew, et al. ”SMPL: A skinned multi-person linear model.” ACM transactions on graphics (TOG) 34.6 (2015): 248.  
[4] SMPL Library implementation : Bogo, Federica, et al. ”Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image.” European Conference on Computer Vision. Springer, Cham, 2016.  
[5] Object detection using tensoflow api tutorial : https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb  
[6] Mask RCNN Model : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md
