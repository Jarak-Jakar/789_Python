# A script to test the quality of the rectification matrix and translation vector derived from a calibration, by using
#  the essential matrix with the calibration points from the two cameras

# Most of the below code has been borrowed from Mihailo Azhar's work

import numpy as np
import argparse
import singleCameraCalibrate
import AuxFunctions as auxF
import OptimisationFunctions as opt
import StereoRectify as rec

## Command line argument parser
parser = argparse.ArgumentParser(description='Filepath to data:')
parser.add_argument('filepathL', help="This is the file path to the left camera  data.")
parser.add_argument('filepathR', help="This is the file path to the right camera  data.")
parser.add_argument('numOfPoints', type=int, help='Number of points to include in calibration')
parser.add_argument('rectifyImageFilepathL', help='This is the file path to the imgae to be rectified')
parser.add_argument('rectifyImageFilepathR', help='This is the file path to the image to be rectified')
parser.add_argument('outputFolder', help='File path to folder to output rectified images')
args = parser.parse_args()

# Initial Parameters
initialParameters = [1, 2]
outputFolder = args.outputFolder
outputRectifyL = outputFolder + "\\RectifiedL.png"
outputRectifyR = outputFolder + "\\RectifiedR.png"
rectifyImageFilepathL = args.rectifyImageFilepathL
rectifyImageFilepathR = args.rectifyImageFilepathR
# Go Pro hero 4
# LeftCamera
pixelWidth = 0.0014
pixelHeight = 0.0014
initialParameters[0] = [pixelWidth, pixelHeight]
initialParameters[1] = [320, 240]

filepathL = args.filepathL
filepathR = args.filepathR
numberOfPoints = args.numOfPoints

## Calculate distortion free parameters of left and right camera
## Returns  [focalLength, translationVector, R, estimatedK1, errorMagnitude,imCoord,sx,XYZ]
resultsL = singleCameraCalibrate.singleCameraCalibrate(filepathL, initialParameters, numberOfPoints, "resultsL.csv")
resultsR = singleCameraCalibrate.singleCameraCalibrate(filepathR, initialParameters, numberOfPoints, "resultsR.csv")

## Find the baseline of unoptimised no distortion
# findBaseline(leftF,leftRot,leftTranslation,rightF,rightRot,rightTranslation,imageCenter):
baselineNoOpt = auxF.findBaseline(resultsL[0], resultsL[2], resultsL[1], resultsR[0], resultsR[2], resultsR[1],
                                  initialParameters[1], pixelHeight)

# naiveTriangulate(imCoordL,imCoordR, focalLength,pixelsize,imageCenter,baseline,R,T,sx):
projectedWC = auxF.naiveTriangulate(resultsL[5], resultsR[5], resultsL[0], initialParameters[0][0],
                                    initialParameters[1], baselineNoOpt, resultsL[2], resultsL[1],
                                    [resultsL[6], resultsR[6]])
error = projectedWC.T - resultsL[7]
averageXNoOpt = np.average(error[:, 0])
averageyNoOpt = np.average(error[:, 1])
averagezNoOpt = np.average(error[:, 2])

## Optimise the camera calibration using undistorted unoptimized camera matrices as initial guess
# optimiseCalibration(focalLength,translationVector,R,k1,data,sx,xyz,imageCenter,pixelDimensions):
outL = opt.optimiseCalibration(resultsL[0], resultsL[1], resultsL[2], resultsL[3], resultsL[5], resultsL[6],
                               resultsL[7], initialParameters[1], initialParameters[0])
outR = opt.optimiseCalibration(resultsR[0], resultsR[1], resultsR[2], resultsR[3], resultsR[5], resultsR[6],
                               resultsR[7], initialParameters[1], initialParameters[0])

# Extract values
optLeftFocal = outL[0]["focalLength"].value
optLeftK1 = outL[0]["k1"].value
optLeftTrans = np.array([[outL[0]["tx"].value], [outL[0]["ty"].value], [outL[0][
                                                                            "tz"].value]])  # np.array([outL["tx"].value,outL["ty"].value,outL["tz"].value])#np.array([[outL["tx"].value],[outL["ty"].value],[outL["tz"].value]])
optLeftSx = [outL[0]["sx"].value]
optLeftR = resultsL[2]

# print("\n Camera 1 post optimisation")
# print("\n Focal length")
# print("before | after")
# print([resultsL[0], optLeftFocal])
# print("\n K1")
# print("before | after")
# print([0, optLeftK1])
# print("\n Translation")
# print("before")
# print(resultsL[1])
# print("after")
# print(optLeftTrans)
# print("\n sx")
# print("before | after")
# print([resultsL[6], optLeftSx])
# print("\n R")
# print(optLeftR)

optRightFocal = outR[0]["focalLength"].value
optRightK1 = outR[0]["k1"].value
optRightTrans = np.array([[outR[0]["tx"].value], [outR[0]["ty"].value], [outR[0][
                                                                             "tz"].value]])  # np.array([outR["tx"].value,outR["ty"].value,outR["tz"].value]) #np.array([[outR["tx"].value],[outR["ty"].value],[outR["tz"].value]])
optRightSx = [outR[0]["sx"].value]
optRightR = resultsR[2]

# print("\n Camera 2 post optimisation")
# print("\n Focal length")
# print("before | after")
# print([resultsR[0], optRightFocal])
# print("\n K1")
# print("before | after")
# print([0, optRightK1])
# print("\n Translation")
# print("before")
# print(resultsR[1])
# print("after")
# print(optRightTrans)
# print("\n sx")
# print("before | after")
# print([resultsR[6], optRightSx])
# print("\n R")
# print(optRightR)

# Find the optimised baseline
# Find baseline of optimised cameras
optBaseline = auxF.findBaseline(optLeftFocal, optLeftR, optLeftTrans, optRightFocal, optRightR, optRightTrans,
                                initialParameters[1], pixelHeight)
# opticalCenters = auxF.findOpticalCenter(optLeftFocal,optLeftR,optLeftTrans,optRightFocal,optRightR,optRightTrans,initialParameters[1])
opticalCenters = auxF.findOpticalCenter(optLeftFocal, optLeftR, optLeftTrans, optRightFocal, optRightR, optRightTrans,
                                        initialParameters[1], pixelHeight)

translationvector = opticalCenters[0] - opticalCenters[1]
Tx = translationvector[0]
Ty = translationvector[1]
Tz = translationvector[2]

translationMatrix = np.array([[0, -1 * Tz, Ty], [Tz, 0, -1 * Tx], [-1 * Ty, Tx, 0]])

# print("\nBaseline")
# print("\n No opt | opt")
# print([baselineNoOpt, optBaseline])

# Comput Optimised back projection
# naiveTriangulate(imCoordL,imCoordR, focalLength,pixelsize,imageCenter,baseline,R,T,sx):
optProjectedWC = auxF.naiveTriangulate(resultsL[5], resultsR[5], optLeftFocal, initialParameters[0][0],
                                       initialParameters[1], optBaseline, optLeftR, optLeftTrans,
                                       [optLeftSx, optRightSx])
error = optProjectedWC.T - resultsL[7]
averageX = np.average(error[:, 0])
averagey = np.average(error[:, 1])
averagez = np.average(error[:, 2])

# print("\n Finding 2D to 3D projection")
# print("\n Average error in X direction")
# print("\n No Opt | Optimised")
# print([averageXNoOpt, averageX])
#
# print("\n Average error in Y direction")
# print("\n No Opt | Optimised")
# print([averageyNoOpt, averagey])
#
# print("\n Average error in Z direction")
# print("\n No Opt | Optimised")
# print([averagezNoOpt, averagez])

### Rectify##################
## Create outputs

rectifiedR = rec.rectifyParameterizeR(optLeftR, opticalCenters[0], opticalCenters[1])
print('\nrectifiedR:')
print(rectifiedR)
# rectifiedR = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# print("\n Rectified Rotation")
# print(rectifiedR)

essentialMatrix = rectifiedR.dot(translationMatrix)
leftM = rec.createKmatrix(optLeftFocal, pixelHeight, initialParameters[1])
rightM = rec.createKmatrix(optRightFocal, pixelHeight, initialParameters[1])

cameraDataL = np.genfromtxt(filepathL, delimiter=',')
cameraDataL = cameraDataL[:numberOfPoints:]
imCoordL = cameraDataL[:, 3:]
imCoordL = np.insert(imCoordL, 2, 1, axis=1)
XYZl = cameraDataL[:, :3]

cameraDataR = np.genfromtxt(filepathR, delimiter=',')
cameraDataR = cameraDataR[:numberOfPoints:]
imCoordR = cameraDataR[:, 3:]
imCoordR = np.insert(imCoordR, 2, 1, axis=1)
XYZr = cameraDataR[:, :3]

print('\nShould be zero:')
for i in range(88):
    shouldbezero = ((np.linalg.inv(rightM).dot(imCoordR[i])).transpose()).dot(essentialMatrix).dot((np.linalg.inv(leftM).dot(imCoordL[i])))
    #shouldbezero = (XYZr[i].transpose()).dot(essentialMatrix).dot(XYZl[i])
    print(shouldbezero)