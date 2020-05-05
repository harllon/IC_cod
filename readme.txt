Source code for "Light Field Video Capture Using a Learning-Based Hybrid Imaging System"

This package is a MATLAB implementation of the algorithm described in:

T.-C. Wang, J.-Y. Zhu, N. K. Kalantari, A. A. Efros, R. Ramamoorthi
"Light Field Video Capture Using a Learning-Based Hybrid Imaging System"
ACM Transaction on Graphics 36, 4, July 2017. 

-------------------------------------------------------------------------
I. OVERVIEW

There are two modes of our code: Test and Hybrid.
1. In Test mode, our algorithm takes in a sequence of LF images, uses all views for the first and the last frame and the central views of all intermediate frames, and reconstructs the missing views for all intermediate frames.
The output is a LF video which consists of all the intermediate frames.
2. In Hybrid mode, our algorithm takes in both a sequence of LF images and a 2D video, where the 2D video serves as the central views for the intermediate frames. Our method then combines them and generates a full LF video.

The code was tested on Ubuntu 14.04, using CUDA 7.0 and cuDNN v3.

-------------------------------------------------------------------------
II. RUNNING CODE

The description for each folder is as follows.

Model:
The trained network model of our system.
Our framework is implemented using a modified version of Caffe. 
A pre-built version for Ubuntu 14.04, CUDA 7.0, cuDNN v3 is included in the "caffe" folder.
Please re-compile it if the pre-built version does not work for you.

Test:
The code to run our algorithm in Test mode.
First please download one test sequence by running "bash data/download_lfv.sh seq01" in this folder.
Then please execute "main.m" and everything should run automatically on the included sequence.
On a machine with Titan X and i7-4930K, it takes approximately 30 seconds to run our code.
If you wish to run on other sequences, please download our full dataset.

Hybrid:
The code to run our algorithm in Hybrid mode.
First please download one hybrid sequence by running "bash data/download_lfv.sh seq01" in this folder.
Then please execute "main.m" and everything should run automatically on the included sequence.
You can choose whether to run only one keyframe or the entire sequence by toggling the "computeAllFrames" parameter.
The computation time for each keyframe depends on how many 2D frames the keyframe spans; usually it takes between 30-50 seconds.
If you wish to run on other sequences, please download our full dataset.


Note: to run the code on your own light fields captured with a Lytro Illum camera,
simply use the Lytro Power Tools (https://www.lytro.com/imaging/power-tools)
to extract the raw light field in png format.

-------------------------------------------------------------------------
IV. VERSION HISTORY

v1.0 - Initial release        (May, 2017)
v1.1 - Separated data files   (May, 2017)
-------------------------------------------------------------------------

If you find any bugs or have comments/questions, please contact 
Ting-Chun Wang at tcwang0509@berkeley.edu.

