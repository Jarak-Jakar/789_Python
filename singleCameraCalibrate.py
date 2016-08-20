import numpy as np
import argparse
import AuxFunctions as auxF
import csv
import OptimisationFunctions as opt

def singleCameraCalibrate(filePath, initialParameters, numberOfPoints, outputFilename):
	## Command line argument parser

	# Team MAH camera parameters
	# TODO: implement read parameter read from file
	pixelWidth = initialParameters[0][0]
	pixelHeight = initialParameters[0][1]
	pixelSize = [pixelWidth, pixelHeight]
	imageCenter = initialParameters[1]

	## Create table of data from csv input file
	cameraData = np.genfromtxt(filePath,delimiter=',')
	numberOfPoints = numberOfPoints
	#Trim to the number of points desired
	cameraData = cameraData[:numberOfPoints:]

	# Camera world coordinates X Y Z
	XYZ = (cameraData[:, :3])
	imCoord = cameraData[:, 3:]

	## Camera calibration step 1
	# Calculate Xd and Yd
	xd = (cameraData[:, 3] - imageCenter[0]) * pixelWidth
	yd = (cameraData[:, 4] - imageCenter[1]) * pixelHeight

	# Calibration step 2
	# Calculate the inverse of M using Monroe-Penrose pseudo inv
	m1 = (yd * cameraData[:, :3].T).T
	m2 = (-xd * cameraData[:, :3].T).T
	my = np.reshape(yd, (-1, 1))
	M = np.concatenate((m1, my, m2), axis=1)
	pseudoInversedM = np.linalg.pinv(M)

	# Calculate our L
	L = np.dot(pseudoInversedM, xd)

	# Calculate inital ty
	ty = 1 / np.sqrt(L[4]**2 + L[5]**2 + L[6]**2)

	# Calculate sx
	sx = ty * np.sqrt(L[0]**2 + L[1]**2 + L[2]**2)

	# Determining the sign of ty
	# Form temporary vector
	r = np.dot(L, abs(ty))
	ty = auxF.get_ty_sign(XYZ, imCoord, xd, yd, r, ty)

	# Recalculate matrix R
	R = np.zeros((3, 3), float)
	R[0][0] = L[0] * ty/sx
	R[0][1] = L[1] * ty/sx
	R[0][2] = L[2] * ty/sx
	R[1][0] = L[4] * ty
	R[1][1] = L[5] * ty
	R[1][2] = L[6] * ty
	tx = L[3] * ty/sx
	R[2][0] = np.cross([R[0][1], R[0][2]], [R[1][1], R[1][2]])
	R[2][1] = np.cross([R[0][2], R[0][0]], [R[1][2], R[1][0]])
	R[2][2] = np.cross([R[0][0], R[0][1]], [R[1][0], R[1][1]])
	lamda = np.sqrt(1 / (R[2][0]**2 + R[2][1]**2 + R[2][2]**2))
	R[2][0] *= lamda
	R[2][1] *= lamda
	R[2][2] *= lamda

	# Step 6 calibration
	# Construct Uy, Uz matrix
	rotate = R[1, :]
	uMat = cameraData[:, :3]
	Uy = np.dot(cameraData[:, :3], R[1, :])
	Uy = Uy + ty
	Uz = np.dot(cameraData[:, :3], R[2, :])

	# Using yd
	y6 = -yd

	# Reshape Uy and y6 into 2D arrays
	Uy = np.reshape(Uy, (1, -1))
	y6 = np.reshape(y6, (1, -1))

	# Construct M from Uy and yd
	M6 = np.concatenate((Uy, y6), axis=0)

	# Construct lhs Mt
	mt = Uz * yd

	# Find f and tz by solving linear equations
	# Like before use Monroe-Penrose psuedo inv
	inversedM6 = np.linalg.pinv(M6)
	solution = np.dot(inversedM6.T, mt)

	# Focal length
	focalLength = solution[0]

	# T vector
	translationVector = np.reshape([tx, ty, solution[1]], (-1, 1))

	## Error calculations
	# Construct Project Matrices
	# Mi
	Mi =[[(1/pixelWidth), 0, imageCenter[0]], [0, (1/pixelHeight), imageCenter[1]], [0, 0, 1]]
	# Mp
	Mp = [[(sx*focalLength), 0, 0, 0], [0, focalLength, 0, 0], [0, 0, 1, 0]]
	# Mr
	Mr = np.concatenate((R, translationVector), axis =1)
	Mr = np.concatenate((Mr, [[0, 0, 0, 1]]), axis=0)

	## Get Pixel Error
	errorMagnitude = auxF.getPixelError(XYZ, imCoord, Mr, Mp, Mi)

	## Get the first distortion coeff
	auxF.estimateK1(errorMagnitude, imCoord, imageCenter, pixelSize)

	## Get Cube Error
	#auxF.getBackProjectError(imCoord, sx, imageCenter[0], imageCenter[1], focalLength, pixelWidth, R, translationVector)

	"""
	## Console Output Display
	print("\nThis is L:\n")
	print(L)

	print('\n Uncertainty scaling factor sx:')
	print(sx)

	print("\nthis is the translation vector T (tx,ty,tx):\n")
	print(translationVector)

	print("\nThis is the rotational matrix R:\n")
	print(R)

	print("\nThis is the effective focal length f:")
	print(focalLength)

	print("\nthis is the solution\n")
	print(solution)
	"""
	estimatedK1 = auxF.estimateK1(errorMagnitude, imCoord, imageCenter, pixelSize)

	## Return vector format
	### focal, T, R, error

	with open(outputFilename, 'w') as fileOut:
		writer = csv.writer(fileOut)
		writer.writerow(['Pre-optimisation'])
		writer.writerow(['f', 'Tx', 'Ty', 'Tz', 'Sx', 'K1'])
		writer.writerow([focalLength, translationVector[0], translationVector[1], translationVector[2], sx, estimatedK1])

		writer.writerow(['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'])
		writer.writerow([R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]])

		writer.writerow(['Xw', 'Yw', 'Zw', 'img x', 'img y', 'Error'])
		for i in range(0,numberOfPoints):
			writer.writerow([XYZ[i][0], XYZ[i][1], XYZ[i][2], imCoord[i][0], imCoord[i][1], errorMagnitude[i]])

		optOut = opt.optimiseCalibration(focalLength, translationVector, R, estimatedK1, imCoord, sx, XYZ, initialParameters[1], initialParameters[0])

		focalLength = optOut[0]["focalLength"].value
		estimatedK1 = optOut[0]["k1"].value
		translationVector[0] = optOut[0]["tx"].value
		translationVector[1] = optOut[0]["ty"].value
		translationVector[2] = optOut[0]["tz"].value
		sx = optOut[0]["sx"].value
		errorMagnitude = optOut[1]

		writer.writerow(['\n\nPost-optimisation'])
		writer.writerow(['f', 'Tx', 'Ty', 'Tz', 'Sx', 'K1'])
		writer.writerow([focalLength, translationVector[0], translationVector[1], translationVector[2], sx, estimatedK1])

		writer.writerow(['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'])
		writer.writerow([R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]])

		writer.writerow(['Xw', 'Yw', 'Zw', 'img x', 'img y', 'Error'])
		for i in range(numberOfPoints):
			writer.writerow([XYZ[i][0], XYZ[i][1], XYZ[i][2], imCoord[i][0], imCoord[i][1], errorMagnitude[i]])
			
		backProjectionErrors = auxF.calculateBackProjection(R, translationVector, focalLength, imCoord, numberOfPoints, XYZ, pixelSize, imageCenter, sx)

		#print("\nbackProjectionErrors = ")
		#print(backProjectionErrors)
		
		writer.writerow(['Xw', 'Yw', 'Zw', 'img x', 'img y', 'Xwerr', 'Ywerr', 'Zwerr'])
		for i in range(numberOfPoints):
			writer.writerow([XYZ[i][0], XYZ[i][1], XYZ[i][2], imCoord[i][0], imCoord[i][1], backProjectionErrors[0][i], backProjectionErrors[1][i], backProjectionErrors[2][i]])
	
	return [focalLength, translationVector, R, estimatedK1, errorMagnitude,imCoord,sx,XYZ]
	## Get the first distortion coeff
	#auxF.estimateK1(errorMagnitude, imCoord, imageCenter, pixelSize)

	## Get Cube Error
	#auxF.getBackProjectError(imCoord, sx, imageCenter[0], imageCenter[1], focalLength, pixelWidth, R, translationVector)
