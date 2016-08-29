"""""
This python file holds auxilary functions that are used for camera calibration
"""""
import os
import numpy as np

# from matplotlib import pyplot as plt
# from scipy.optimize import minimize, rosen_der
# import skimage
# from skimage import data
# from skimage import io
# io.use_plugin('matplotlib')
# from skimage.feature import corner_harris, corner_subpix, corner_peaks


"""
This function finds the furthest point from the center by summing across their squared values of x,y
Input:
XYZ -> table of measured points in 3D space
imCoord -> table of image pixel locations of points
xd -> X coordinate of distorted image point
yd -> Y coordinate of distorted image points
R -> 3x3 rotation matrix
ty -> Initial estimate of ty

Output:
ty -> Magnitude and sign of Y component of translation vector
"""


def get_ty_sign(xyz, imCoord, xd, yd, R, ty):
    # Find point furthest from image principal point
    sumSquared = xd ** 2 + yd ** 2
    indexLargest = np.argmax(sumSquared, 0)
    furthestPoint = xyz[indexLargest, :]
    furthestPointIm = imCoord[indexLargest, :]
    # caluclate Xt Yt
    Xc = R[0] * furthestPoint[0] + R[1] * furthestPoint[1] + R[3]  # R[3] is tx
    Yc = R[4] * furthestPoint[0] + R[5] * furthestPoint[1] + ty

    # Determine sign
    if (np.copysign(1, Xc) == np.copysign(1, furthestPointIm[0]) or np.copysign(1, Yc) == np.copysign(1,
                                                                                                      furthestPointIm[
                                                                                                          1])):
        return ty
    return -ty


"""
This function calculates the pixel error for points in 3D world coordinates projected to image plane
Input:
XYZ -> table of measured points in 3D space
imCoord -> table of image pixel locations of points
Me -> Rigid body transofrmation matrix. Note this is 4x4 RT in homogenous coordinates
Mp -> Perspective transform matrix
Mi -> Image pixel to mm transform matrix

Output: Array of pixel errors for each point
"""


def getPixelError(XYZ, imCoord, Me, Mp, Mi):
    ones = np.reshape(np.ones(XYZ.shape[0]), (1, -1))
    homogenous_XYZ = np.concatenate((XYZ.T, ones), axis=0)

    # Pixel wise
    kxeYe = np.dot(Mi, np.dot(Mp, np.dot(Me, homogenous_XYZ)))
    Xe = kxeYe[0, :] / kxeYe[2, :]
    Ye = kxeYe[1, :] / kxeYe[2, :]

    ex = (Xe - imCoord[:, 0])
    ey = (Ye - imCoord[:, 1])
    # Calculate error magnitude
    errorMagnitude = np.sqrt(ex ** 2 + ey ** 2)
    # Calculate Average and Std
    print("\n Average Error magnitude")
    print(np.average(errorMagnitude))
    print("\n Stdev ")
    print(np.std(errorMagnitude))
    return errorMagnitude


"""
This function calculates an estimate for the first coefficient of radial distortion
Input:
error -> array of differences between estimated pixel locations and measure locations for points
imCoord -> table of image pixel locations of points
pixelSize -> array of pixel sizes for image sensor x and y
"""


def estimateK1(error, imCoord, imageCenter, pixelSize):
    rd = np.sqrt(
        ((imCoord[:, 0] - imageCenter[0]) * pixelSize[0]) ** 2 + ((imCoord[:, 1] - imageCenter[1]) * pixelSize[1]) ** 2)
    eK1 = (error * pixelSize[0]) / (np.power(rd, 3))
    averageK1 = np.average(eK1)
    return averageK1


# print("\n Estimated K1")
# print(averageK1)

"""
This function obtains the subpixel location of corners in a chessboard

def detectCornerSubpix(imageFilepath):
	##Testing subpixel obtaining
	leftimage = io.imread(imageFilepath,True)
	coords = corner_peaks(corner_harris(leftimage),min_distance=5)
	#coords_subpix = corner_subpix(leftimage, coords,window_size=13)
	f1 = plt.figure()
	io.imshow(leftimage)
	plt.plot(coords[:, 1], coords[:, 0], '.b', markersize=7)
	plt.show()
"""

"""
This function calculates the baseline between 2 cameras as the difference between their optical centers
Input:
leftF : left focal length
leftRot : left rotation matrix
leftTranslation : left translation vector
rightF : right focal length
rightRot : right rotation matrix
rightTranslation : right translation vector
imageCenter : center of image [center of X, center of Y]
Output:
baseline: Distance between optical centers of two cameras
"""


def findBaseline(leftF, leftRot, leftTranslation, rightF, rightRot, rightTranslation, imageCenter, pixelSize):
    """
	#assemble left RT
	leftRT =np.concatenate((leftRot, leftTranslation), axis=1)
	rightRT =np.concatenate((rightRot, rightTranslation), axis=1)

	# Form projection matrix
	leftProjection= np.array([[leftF, 0, imageCenter[0]],[0, leftF, imageCenter[1]], [0,0,1]])
	rightProjection= np.array([[rightF, 0, imageCenter[0]],[0, rightF, imageCenter[1]], [0,0,1]])

	# Form Qq
	leftQq = np.dot(leftProjection,leftRT)
	rightQq = np.dot(rightProjection,rightRT)
	leftQ = leftQq[:,:3]
	leftq = leftQq[:,3]
	rightQ = rightQq[:,:3]
	rightq = rightQq[:,3]

	leftOpticalCenter = np.dot(-1*np.linalg.inv(leftQ),leftq)
	rightOpticalCenter = np.dot(-1*np.linalg.inv(rightQ),rightq)
	"""
    opticalCenters = findOpticalCenter(leftF, leftRot, leftTranslation, rightF, rightRot, rightTranslation, imageCenter, pixelSize)
    baseline = opticalCenters[1] - opticalCenters[0]
    baseline = np.sqrt(baseline[0] ** 2 + baseline[1] ** 2 + baseline[2] ** 2)
    # print("\nBaseline")
    # print(baseline)
    return baseline


def findOpticalCenter(leftF, leftRot, leftTranslation, rightF, rightRot, rightTranslation, imageCenter, pixelSize):
    # assemble left RT
    leftRT = np.concatenate((leftRot, leftTranslation), axis=1)
    rightRT = np.concatenate((rightRot, rightTranslation), axis=1)

    # Form projection matrix
    leftProjection = np.array([[(leftF / pixelSize), 0, imageCenter[0]], [0, (leftF / pixelSize), imageCenter[1]], [0, 0, 1]])
    rightProjection = np.array([[(rightF / pixelSize), 0, imageCenter[0]], [0, (rightF / pixelSize), imageCenter[1]], [0, 0, 1]])

    # Form Qq
    leftQq = np.dot(leftProjection, leftRT)
    rightQq = np.dot(rightProjection, rightRT)
    leftQ = leftQq[:, :3]
    leftq = leftQq[:, 3]
    rightQ = rightQq[:, :3]
    rightq = rightQq[:, 3]

    leftOpticalCenter = np.dot(-1 * np.linalg.inv(leftQ), leftq)
    rightOpticalCenter = np.dot(-1 * np.linalg.inv(rightQ), rightq)

    return [leftOpticalCenter, rightOpticalCenter]


"""
This function performs a naive triangulation using camera extrinsic and intrinsic parameters
and corresponding image locations for 2 cameras where one camera is designated main camera

Input:
imCoordL : Image coordinates of calibration points in pixels for left camera
imCoordR : Image coordinates of calibration points in pixels for right camera
focalLength : focal length
pixelsise: Pixel size assuming square pixels
imageCenter : center of image [center of X, center of Y]
baseline : Distance between the optical centers of left and right camera
R: rotation matrix of left camera
T: translation vector of left camera
sx: Horizational scaling factor [sx of left camera, sx of right camera]
Output: projected 3D world coordinates
"""


def naiveTriangulate(imCoordL, imCoordR, focalLength, pixelsize, imageCenter, baseline, R, T, sx):
    xld = (pixelsize * (imCoordL[:, 0] - imageCenter[0])) / sx[0]
    xrd = (pixelsize * (imCoordR[:, 0] - imageCenter[0])) / sx[1]
    yld = pixelsize * (imCoordL[:, 1] - imageCenter[1])

    z = focalLength * baseline / (xld - xrd)
    x = xld * (z / focalLength)
    y = yld * (z / focalLength)
    projectPointCamera = np.array([x, y, z])
    Rinv = np.linalg.inv(R)
    translated = projectPointCamera - T
    projectWorld = np.dot(Rinv, translated)
    return projectWorld


def calculateBackProjection(R, transV, focalLength, imCoord, numPoints, xyz, pixelDimensions, imageCenter, sx):
    tx = transV[0][0]
    ty = transV[1][0]
    tz = transV[2][0]

    xw_proj = 0
    yw_proj = 0
    zw_proj = 0
    xcamera = (pixelDimensions[0]) * (imCoord[:, 0] - imageCenter[0])  # sx[0] *
    ycamera = (pixelDimensions[1]) * (imCoord[:, 1] - imageCenter[1])
    results = []
    for i in range(0, numPoints):
        if xyz[i][0] == 0:
            # calculate alpha
            xw_proj = 0
            alpha = -tx / (R[0][0] * xcamera[i] + R[0][1] * ycamera[i] - R[0][2] * focalLength)
            yw_proj = ty + alpha * (R[1][0] * xcamera[i] + R[1][1] * ycamera[i] - R[1][2] * focalLength)
            zw_proj = tz + alpha * (R[2][0] * xcamera[i] + R[2][1] * ycamera[i] - R[2][2] * focalLength)
        if xyz[i][1] == 0:
            yw_proj = 0
            alpha = -ty / (R[1][0] * xcamera[i] + R[1][1] * ycamera[i] - R[1][2] * focalLength)
            xw_proj = tx + alpha * (R[0][0] * xcamera[i] + R[0][1] * ycamera[i] - R[0][2] * focalLength)
            zw_proj = tz + alpha * (R[2][0] * xcamera[i] + R[2][1] * ycamera[i] - R[2][2] * focalLength)
        results.append([xw_proj, yw_proj, zw_proj])

    # Calculate error
    resultsnp = np.array(results)
    errorsXw = resultsnp[:, 0] - xyz[:, 0]
    errorsYw = resultsnp[:, 1] - xyz[:, 1]
    errorsZw = resultsnp[:, 2] - xyz[:, 2]
    meanErrorMag = np.mean(np.sqrt(errorsXw ** 2 + errorsYw ** 2 + errorsZw ** 2))
    print("\nmean error mag")
    print(meanErrorMag)
    return  resultsnp


"""
This function paramterises R into a 3x1 vector following the rodriguez formula
"""


def forwardParameteriseR(R):
    magnitude_W = np.arccos((np.trace(R) - 1) / 2)
    differenceR = [[R[2][1] - R[1][2]], [R[0][2] - R[2][0]], [R[1][0] - R[0][1]]]
    parameteriseR = np.dot((1 / (2 * np.sin(magnitude_W))), differenceR)
    return parameteriseR, magnitude_W


def backwardParameteriseR(W, magnitude_W):
    # definedW = np.array([[0,-W[2],W[1]],[W[2],0,-W[0]],[-W[1],W[0],0]])
    definedW = np.array([[0, -W[2], W[1]],
                         [W[2], 0, -W[0]],
                         [-W[1], W[0], 0]])
    theta = np.sqrt(W[0] ** 2 + W[1] ** 2 + W[2] ** 2)
    # = eye(3) + (sin(theta)/theta)*omega + ((1-cos(theta))/theta^2)*(omega*omega);
    R = np.identity(3) + (np.sin(theta) / theta) * definedW + ((1 - np.cos(theta)) / (theta ** 2)) * (
    definedW * definedW)
    return R
