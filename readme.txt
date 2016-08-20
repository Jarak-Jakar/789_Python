StereoCalibrate.py readmefile
Author: Mihailo Azhar
Email: mazh003@gmail.com

//------------------
System Requirements:
//------------------

Python 3.5 or greater
May be compatible with 3.4

//----------------
Packages and libraries
//----------------
To run the script successfully, the following packages are requried in dependency order:

	* LAPACK and BLAS or ATLAS
	* numpy or numpy+mkl (includes LAPACK and BLAS)
	* scipy
	* lmfit

//----------------
LAPACK and BLAS
//----------------
This python script requrires LAPACK and BLAS to run. These should be installed before attempting to pip install scipy.

//------------------
Python libraries
//------------------

--------------------
Numpy or numpy + mkl
--------------------
Numpy- The numpy library is required to run this script. The numpy+mkl provides
a compilation optimization for intel chips see: https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl
and includes LAPACK and BLAS.

Install command:
	To install numpy run the command in the command prompt/terminal:
	pip install numpy

Install numpy+mkl:
	Precompiled binaries for python+mkl distributions can be found here: http://www.lfd.uci.edu/~gohlke/pythonlibs/
	Download numpy-1.10.4+mkl-cp34-cp34m-win32.whl.
	On the command line type the command:
	pip install C:/some-dir/numpy-1.10.4+mkl-cp34-cp34m-win32.whl

-----------------
Scipy
-----------------
NOTE: Your systems needs LAPACK and BLAS or ATLAS in order to install scipy via pip through a remote repository.

Requires: LAPACK and BLAS on your machine or numpy+mkl

Install command:
	To install scipy run the command in the command prompt/terminal:
	pip install scipy

Alternative Install:
	Install scipy from prebuilt binaries whl file. Download scipy-0.17.1-cp34-cp34m-win32.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/
	On the command line type the command:
	pip install C:/some-dir/scipy-0.17.1-cp34-cp34m-win32.whl

----------------
lmfit
----------------
This package is a optimization package built on scipy that allows for easy handling of objective functions and minimisation methods.
 
Requires: scipy

Install command: 
	pip install lmfit

Alternative install: 
	Proceed to http://cars9.uchicago.edu/software/python/lmfit/installation.html and follow steps

----------------
Scikit-image
----------------
This package is a image processing library built on scipy that allows for easy handling of IO for images.
 
Requires: scipy

Install command: 
	pip install scikit-image
	
Note: An internet connection is required to download the package (small download)
Troubleshooting:

If pip is not enabled, reinstall your python version with pip tools installed.

//--------------------
Script input arguments for StereoCalibrateAndRectify
//--------------------
filepathL - file path to dataset for left camera. NOTE: the dataset should be in csv format, see example files
filepathR - file path to dataset for right camera. NOTE: the dataset should be in csv format, see example files
numOfPoints - number of points you wish to calibrate
rectifyImageFilepathL - This is the file path to the imgae to be rectified
rectifyImageFilepathR - This is the file path to the image to be rectified
outputFolder - File path to folder to output rectified images

//--------------------
Script input arguments for StereoMatching
//--------------------
filepathL - This is the file path to the left camera  data.
filepathR - This is the file path to the right camera  data.
outputFolder - The output folder for the disparity image.
windowSize - type=int - Half size of matching window.
disparitySize - type=int - Disparity estimate to search through.

//--------------------
Script input arguments for StereoCalibrate
//--------------------
filepathL - file path to dataset for left camera. NOTE: the dataset should be in csv format, see example files
filepathR - file path to dataset for right camera. NOTE: the dataset should be in csv format, see example files
numOfPoints - number of points you wish to calibrate

//---------------
Running the tool
//---------------
From command line windows/linux/unix
> python StereoCalibrate.py <filepathL> <filepathR> <numOfPoints>

example:
> python StereoCalibrate.py C:\User1\CalibrationDataset\CalibrationDataLeft.csv C:\User1\CalibrationDataset\CalibrationDataRight.csv 18

//-------------
Troubleshooting
//-------------
Ensure your system's environment variables point to the correct version of python. You
can check this using the following command in the command prompt/terminal:

python --version

NOTE:
Most unix and linux distros have Python 2.4 installed by default.
Installing 3.4 on this system maybe result in using the command python34 or python3.

To change the python environment variable on Windows please google it.