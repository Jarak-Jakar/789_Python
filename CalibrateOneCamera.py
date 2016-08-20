import singleCameraCalibrate
import csv

pixelWidth = 0.0014
pixelHeight = 0.0014
imageCentre = [1296, 972]
numPoints = 30
inputParams = [1, 2] 
inputParams[0] = [pixelWidth, pixelHeight]
inputParams[1] = imageCentre

output = singleCameraCalibrate.singleCameraCalibrate("SingleCamCalib.csv", inputParams, numPoints, "SingleCamCalibResults.csv")

#fileOut = open("SingleCamCalibResults.csv", "wb")
#writer = csv.writer(fileOut)
#writer.writerows(output)

#print(output)
