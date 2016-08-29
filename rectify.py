import cv2
import numpy as np
import StereoRectify as rec
import AuxFunctions as auxF
import singleCameraCalibrate
import OptimisationFunctions as opt
import math
import timeit


def rectifyImagesHyun(leftImage, leftFocalLength, leftCameraMatrix, leftRotationMatrix, leftTranslationVector, rightImage,
                      rightFocalLength,
                      rightCameraMatrix, rightRotationMatrix, rightTranlsationMatrix, imageCentre, pixelSize):
    opticalCenters = auxF.findOpticalCenter(leftFocalLength, leftRotationMatrix, leftTranslationVector,
                                            rightFocalLength, rightRotationMatrix,
                                            rightTranlsationMatrix, imageCentre, pixelSize)

    rectifiedR = rec.rectifyParameterizeR(leftRotationMatrix, opticalCenters[0], opticalCenters[1])

    # rectifiedR = np.array([[9.9790417658222752e-001, -2.3270560176052738e-002,
    # -6.0379925379543391e-002], [3.0579117318157156e-002,
    # 9.9192377895663997e-001, 1.2309400604587155e-001],
    # [5.7027817280594743e-002, -1.2468238756725582e-001,
    # 9.9055647506174005e-001]])

    print("\n Rectified Rotation")
    print(rectifiedR)

    kNew = rec.calcNewK(leftCameraMatrix, rightCameraMatrix)

    leftCoeffMat = leftCameraMatrix.dot(np.transpose(rectifiedR).dot(np.linalg.inv(kNew)))
    rightCoeffMat = rightCameraMatrix.dot(np.transpose(rectifiedR).dot(np.linalg.inv(kNew)))

    print("\nleftCoeffMat = ")
    print(leftCoeffMat)

    print("\nrightCoeffMat = ")
    print(rightCoeffMat)

    #print("\nAbout to do the rectification of the left image")

    # rectify images

    #t = timeit.Timer(stmt=lambda: rectifyimagehyun(leftImage, leftCoeffMat))

    #print(t.repeat(3,1))

    rectLeftImage = rectifyimagehyun(leftImage, leftCoeffMat)

    #print("\nAbout to do the rectification of the right image")

    rectRightImage = rectifyimagehyun(rightImage, rightCoeffMat)

    #print("Finished the rectifications, about to return from rectify")

    return rectLeftImage, rectRightImage


def rectifyimagehyun(image, coefficientMatrix):
    c1 = coefficientMatrix[0:2, 0:1]
    c2 = coefficientMatrix[0:2, 1:2]
    rectImage = np.zeros(image.shape)
    oldCoords = np.zeros((2,1)) - (c2 - (image.shape[1] * c1))
    blInter = createbilinInterpolator(image)

    # testCoords = np.array([320.4, 239.8])
    # t = timeit.Timer(stmt=lambda: blInter(testCoords))
    #
    # print(t.repeat(5, 10))

    # Do the calculation of the new coordinates
    for y in range(image.shape[1]):
        newCoords = oldCoords + (c2 - ((image.shape[0] - 1) * c1))
        #rectImage[0][y] = bilinInterp(image, newCoords)
        rectImage[0][y] = blInter(newCoords)
        np.copyto(oldCoords, newCoords)
        for x in range(1, image.shape[0]):
            newCoords = oldCoords + c1
            #rectImage[x][y] = bilinInterp(image, newCoords)
            rectImage[x][y] = blInter(newCoords)
            np.copyto(oldCoords, newCoords)

    return rectImage.astype(np.uint8)


# Code for this blatantly copied from Gee's CS 773 Rectification slides
def createbilinInterpolator(image):
    colourExtractor = createColourExtractor(image)
    # testCoords = np.array([320.4, 239.8])
    # t = timeit.Timer(stmt=lambda: colourExtractor(testCoords[0], testCoords[1]))
    #
    # print(t.repeat(5, 100))

    def bilinInterp(coords):
        x1 = math.floor(coords[0])
        x2 = x1 + 1
        y1 = math.floor((coords[1]))
        y2 = y1 + 1

        # colour1 = extractColour(image, x1, y1) * (x2 - coords[0]) * (y2 - coords[1])
        # colour2 = extractColour(image, x2, y1) * (coords[0] - x1) * (y2 - coords[1])
        # colour3 = extractColour(image, x1, y2) * (x2 - coords[0]) * (coords[1] - y1)
        # colour4 = extractColour(image, x2, y2) * (coords[0] - x1) * (coords[1] - y1)

        colour1 = colourExtractor(x1, y1) * (x2 - coords[0]) * (y2 - coords[1])
        colour2 = colourExtractor(x2, y1) * (coords[0] - x1) * (y2 - coords[1])
        colour3 = colourExtractor(x1, y2) * (x2 - coords[0]) * (coords[1] - y1)
        colour4 = colourExtractor(x2, y2) * (coords[0] - x1) * (coords[1] - y1)

        return colour1 + colour2 + colour3 + colour4
    return bilinInterp


def createColourExtractor(image):
    def extractColour(x, y):
        if x < 0 or x >= image.shape[0]:
            return 0
        if y < 0 or y >= image.shape[1]:
            return 0
        return image[x][y]
    return extractColour
