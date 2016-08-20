import numpy as np
import argparse
import AuxFunctions as auxF
import singleCameraCalibrate
import OptimisationFunctions as opt
import StereoRectify as rec
import rectify
import cv2
from skimage import io, util
import skimage as skimage
import matplotlib.pyplot as plt
import math as math
import scipy as scipy
from scipy import misc


## Command line argument parser
parser = argparse.ArgumentParser(description='Filepath to data:')
parser.add_argument('filepathL', help="This is the file path to the left camera  data.")
parser.add_argument('filepathR', help="This is the file path to the right camera  data.")
parser.add_argument('numOfPoints', type=int, help='Number of points to include in calibration')
parser.add_argument('rectifyImageFilepathL',help='This is the file path to the imgae to be rectified')
parser.add_argument('rectifyImageFilepathR',help='This is the file path to the image to be rectified')
parser.add_argument('outputFolder',help='File path to folder to output rectified images')
args = parser.parse_args()

# Initial Parameters
initialParameters = [1,2]
outputFolder = args.outputFolder
outputRectifyL = outputFolder + "\\RectifiedL.png"
outputRectifyR = outputFolder + "\\RectifiedR.png"
rectifyImageFilepathL = args.rectifyImageFilepathL
rectifyImageFilepathR = args.rectifyImageFilepathR
#Go Pro hero 4
#LeftCamera
pixelWidth = 0.0014
pixelHeight = 0.0014
initialParameters[0] = [pixelWidth, pixelHeight]
initialParameters[1] = [320, 240]

filepathL = args.filepathL
filepathR = args.filepathR
numberOfPoints = args.numOfPoints

## Calculate distortion free parameters of left and right camera
## Returns  [focalLength, translationVector, R, estimatedK1, errorMagnitude,imCoord,sx,XYZ]
resultsL = singleCameraCalibrate.singleCameraCalibrate(filepathL, initialParameters,numberOfPoints)
resultsR = singleCameraCalibrate.singleCameraCalibrate(filepathR, initialParameters,numberOfPoints)

## Find the baseline of unoptimised no distortion
#findBaseline(leftF,leftRot,leftTranslation,rightF,rightRot,rightTranslation,imageCenter):
baselineNoOpt = auxF.findBaseline(resultsL[0],resultsL[2],resultsL[1],resultsR[0],resultsR[2],resultsR[1],initialParameters[1])

#naiveTriangulate(imCoordL,imCoordR, focalLength,pixelsize,imageCenter,baseline,R,T,sx):
projectedWC = auxF.naiveTriangulate(resultsL[5],resultsR[5], resultsL[0],initialParameters[0][0],initialParameters[1],baselineNoOpt,resultsL[2],resultsL[1],[resultsL[6],resultsR[6]])
error = projectedWC.T - resultsL[7]
averageXNoOpt = np.average(error[:,0])
averageyNoOpt = np.average(error[:,1])
averagezNoOpt = np.average(error[:,2])

## Optimise the camera calibration using undistorted unoptimized camera matrices as initial guess
#optimiseCalibration(focalLength,translationVector,R,k1,data,sx,xyz,imageCenter,pixelDimensions):
outL = opt.optimiseCalibration(resultsL[0],resultsL[1],resultsL[2],resultsL[3],resultsL[5],resultsL[6],resultsL[7],initialParameters[1],initialParameters[0])
outR = opt.optimiseCalibration(resultsR[0],resultsR[1],resultsR[2],resultsR[3],resultsR[5],resultsR[6],resultsR[7],initialParameters[1],initialParameters[0])

#Extract values
optLeftFocal = outL["focalLength"].value
optLeftK1 = outL["k1"].value
optLeftTrans =np.array([[outL["tx"].value],[outL["ty"].value],[outL["tz"].value]]) #np.array([outL["tx"].value,outL["ty"].value,outL["tz"].value])#np.array([[outL["tx"].value],[outL["ty"].value],[outL["tz"].value]])
optLeftSx = [outL["sx"].value]
optLeftR = resultsL[2]

print("\n Camera 1 post optimisation")
print("\n Focal length")
print("before | after")
print([resultsL[0], optLeftFocal])
print("\n K1")
print("before | after")
print([0, optLeftK1])
print("\n Translation")
print("before")
print(resultsL[1])
print("after")
print(optLeftTrans)
print("\n sx")
print("before | after")
print([resultsL[6], optLeftSx])
print("\n R")
print(optLeftR)


optRightFocal = outR["focalLength"].value
optRightK1 = outR["k1"].value
optRightTrans =np.array([[outR["tx"].value],[outR["ty"].value],[outR["tz"].value]])#np.array([outR["tx"].value,outR["ty"].value,outR["tz"].value]) #np.array([[outR["tx"].value],[outR["ty"].value],[outR["tz"].value]])
optRightSx = [outR["sx"].value]
optRightR = resultsR[2]

print("\n Camera 2 post optimisation")
print("\n Focal length")
print("before | after")
print([resultsR[0], optRightFocal])
print("\n K1")
print("before | after")
print([0, optRightK1])
print("\n Translation")
print("before")
print(resultsR[1])
print("after")
print(optRightTrans)
print("\n sx")
print("before | after")
print([resultsR[6], optRightSx])
print("\n R")
print(optRightR)

#Find the optimised baseline
#Find baseline of optimised cameras
optBaseline = auxF.findBaseline(optLeftFocal,optLeftR,optLeftTrans,optRightFocal,optRightR,optRightTrans,initialParameters[1])
#opticalCenters = auxF.findOpticalCenter(optLeftFocal,optLeftR,optLeftTrans,optRightFocal,optRightR,optRightTrans,initialParameters[1])
opticalCenters = auxF.findOpticalCenter((optLeftFocal/pixelHeight),optLeftR,optLeftTrans,(optRightFocal/pixelWidth),optRightR,optRightTrans,initialParameters[1])

print("\nBaseline")
print("\n No opt | opt")
print([baselineNoOpt,optBaseline])

#Comput Optimised back projection
#naiveTriangulate(imCoordL,imCoordR, focalLength,pixelsize,imageCenter,baseline,R,T,sx):
optProjectedWC = auxF.naiveTriangulate(resultsL[5], resultsR[5], optLeftFocal,initialParameters[0][0],initialParameters[1],optBaseline,optLeftR,optLeftTrans,[optLeftSx,optRightSx])
error =optProjectedWC.T - resultsL[7]
averageX = np.average(error[:,0])
averagey = np.average(error[:,1])
averagez = np.average(error[:,2])

print("\n Finding 2D to 3D projection")
print("\n Average error in X direction")
print("\n No Opt | Optimised")
print([averageXNoOpt,averageX])

print("\n Average error in Y direction")
print("\n No Opt | Optimised")
print([averageyNoOpt,averagey])

print("\n Average error in Z direction")
print("\n No Opt | Optimised")
print([averagezNoOpt,averagez])


### Rectify##################
## Create outputs

rectifiedR = rec.rectifyParameterizeR(optLeftR,opticalCenters[0],opticalCenters[1])
#rectifiedR = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
print("\n Rectified Rotation")
print(rectifiedR)
kLeft = rec.createKmatrix(optLeftFocal,pixelWidth,initialParameters[1])
#kLeft = np.array([[optLeftFocal, 0, initialParameters[1][0]], [0,optLeftFocal,initialParameters[1][1]],[0,0,1]])
kRight = rec.createKmatrix(optRightFocal,pixelWidth,initialParameters[1])
#kRight = np.array([[optRightFocal, 0, initialParameters[1][0]], [0,optRightFocal,initialParameters[1][1]],[0,0,1]])
kAverage = (kLeft + kRight) / 2

leftUnrectImage = cv2.imread(rectifyImageFilepathL)
rightUnrectImage = cv2.imread(rectifyImageFilepathR)

trevorsR = np.array([[9.9790417658222752e-001, -2.3270560176052738e-002,
    -6.0379925379543391e-002], [3.0579117318157156e-002,
    9.9192377895663997e-001, 1.2309400604587155e-001],
    [5.7027817280594743e-002, -1.2468238756725582e-001,
    9.9055647506174005e-001]])

trevorsT = np.array([[-9.9678007841280525e-001], [-4.1303518227517835e-002], [6.8727684824901214e-002]])

# rectifiedImages = rectify.rectifyImagesHyun(leftUnrectImage, optLeftFocal, kLeft, optLeftR, optLeftTrans, rightUnrectImage,
#                                              optRightFocal, kRight, optRightR, optRightTrans, initialParameters[1])

# Take the OpenCV approach

distcoeffs = np.zeros(5)

outR1, outR2, outP1, outP2, outQ, junk1, junk2 = cv2.stereoRectify(cameraMatrix1=kLeft, distCoeffs1=distcoeffs, cameraMatrix2=kRight,
                                                     distCoeffs2=distcoeffs, imageSize=(640, 480), #np.array((leftUnrectImage.shape[0],leftUnrectImage.shape[1])),
                                                     R=trevorsR, T=trevorsT)

print("distcoeffs = ")
print(distcoeffs)

map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=kAverage, distCoeffs=distcoeffs, R=outR1,
                                         newCameraMatrix=outP1, size=(640, 480), m1type=cv2.CV_16SC2)

leftRectImage = cv2.remap(src=leftUnrectImage, map1=map1, map2=map2, interpolation=cv2.INTER_LINEAR)
rightRectImage = cv2.remap(src=rightUnrectImage, map1=map1, map2=map2, interpolation=cv2.INTER_LINEAR)

# cv2.imwrite("leftRectifiedImageHyun.jpg", rectifiedImages[0])
# cv2.imwrite("rightRectifiedImageHyun.jpg", rectifiedImages[1])

cv2.imwrite("leftRectifiedImageOCV.jpg", leftRectImage)
cv2.imwrite("rightRectifiedImageOCV.jpg", rightRectImage)

leftRectImageGray = cv2.cvtColor(leftRectImage, cv2.COLOR_BGR2GRAY)
rightRectImageGray = cv2.cvtColor(rightRectImage, cv2.COLOR_BGR2GRAY)

stereomatcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparityimage = stereomatcher.compute(rightRectImageGray, leftRectImageGray)

#cv2.imshow("dispa", disparityimage)
#cv2.waitKey(0)

cv2.imwrite("disparitymap.jpg", disparityimage)

# pMatLeftOld = rec.createProjectionMatrix(kLeft,optLeftR,optLeftTrans)
# pMatRightOld = rec.createProjectionMatrix(kRight,optRightR,optRightTrans)
# #print("\nOld LeftProbection")
# #print(pMatLeftOld)
# #print("\nOld RightProbection")
# #print(pMatRightOld)
# kNew = rec.calcNewK(kLeft,kRight)
# newPLeft = rec.createNewPMatrix(kNew,rectifiedR,opticalCenters[0])
# newPRight = rec.createNewPMatrix(kNew,rectifiedR,opticalCenters[1])
#
# print("\nNew Left Projection Matrix")
# print(newPLeft)
# print("\nNew Right Projection Matrix")
# print(newPRight)
#
# homographyLeft = rec.createHomography(pMatLeftOld,newPLeft)
# homographyRight = rec.createHomography(pMatRightOld,newPRight)
#
# leftImageFilepath = rectifyImageFilepathL
# rightImageFilepath = rectifyImageFilepathR
#
#
# leftImage = (io.imread(leftImageFilepath))#.astype("uint8")
#
# rightImage = (io.imread(rightImageFilepath))#.astype("uint8")#(float)
# print("\nNew K Matrix")
# print(kNew)
#
# print("\n H1")
# print(homographyLeft)
# print("\n H2")
# print(homographyRight)
#
# rectImage = np.zeros(leftImage.shape)
# imageWidth = leftImage.shape[1]
# imageHeight = leftImage.shape[0]
# inversedH1 = np.linalg.inv(homographyLeft)
# inversedH2 = np.linalg.inv(homographyRight)
#
# print("\b RECTIFYING LEFT IMAGE")
# image = leftImage
# for row in range(0,image.shape[0]):
# 	for col in range(0,image.shape[1]):
# 		coordinate = np.array([[col],[row],[1]])
# 		homogenousCood = np.dot(inversedH1,coordinate)
# 		homographyCol = int(np.floor(homogenousCood[0]))
# 		homographyRow = int(np.floor(homogenousCood[1]))
# 		if(homographyRow >= 0 and homographyCol >=0 and homographyCol < imageWidth and homographyRow < imageHeight):
# 			rectImage[row][col] = image[(imageHeight-1) - homographyRow][(imageWidth -1) - homographyCol]
#
# rectImage = rectImage.astype("uint8")
#
#
# skimage.io.imsave(outputRectifyL,rectImage)
#
# print("\b RECTIFYING RIGHT IMAGE")
# rectImageRight = np.zeros(leftImage.shape)
# for row in range(0,rightImage.shape[0]):
# 	for col in range(0,rightImage.shape[1]):
# 		coordinate = np.array([[col],[row],[1]])
# 		homogenousCood = np.dot(inversedH2,coordinate)
# 		homographyCol = int(np.floor(homogenousCood[0]))
# 		homographyRow = int(np.floor(homogenousCood[1]))
# 		if(homographyRow >= 0 and homographyCol >=0 and homographyCol < imageWidth and homographyRow < imageHeight):
# 			rectImageRight[row][col] = rightImage[(imageHeight-1) - homographyRow][(imageWidth -1) - homographyCol]
#
# rectImageRight = rectImageRight.astype("uint8")
# skimage.io.imsave(outputRectifyR,rectImageRight)
