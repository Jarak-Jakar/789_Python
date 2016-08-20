import AuxFunctions
from lmfit import minimize, Parameters, fit_report
import numpy as np
"""
This is the objective function that we are going to minimuse
The fixed input parameters are R,
The params to be optimised are Ty,Tx,Tz, focalLength, K1
Data is the measure pixel locations
"""
def residual(params,data,R,xyz,imageCenter,pixelDimensions):
	ty = np.array([params['ty'].value])
	tx = np.array([params['tx'].value])
	tz = np.array([params['tz'].value])
	focalLength = params['focalLength'].value
	k1 = params['k1'].value
	sx = params['sx'].value

	## Find pixel locations
	#TODO extract into a separate method
	ones = np.reshape(np.ones(xyz.shape[0]), (1, -1))
	homogenous_XYZ = np.concatenate((xyz.T, ones), axis=0)
	translationVector = np.array([tx,ty,tz])
	pixelWidth = pixelDimensions[0]
	pixelHeight = pixelDimensions[1]
	Mi =[[(1/pixelWidth), 0, imageCenter[0]], [0, (1/pixelHeight), imageCenter[1]], [0, 0, 1]]
	# Mp
	Mp = [[(sx*focalLength), 0, 0, 0], [0, focalLength, 0, 0], [0, 0, 1, 0]]
	# Mr
	Mr = np.concatenate((R, translationVector), axis=1)
	Mr = np.concatenate((Mr, [[0, 0, 0, 1]]), axis=0)
	mrpResult = np.dot(Mp, np.dot(Mr, homogenous_XYZ))

	# X and Y undistorted camera coordinate
	xdistCalc =mrpResult[0, :] / mrpResult[2, :]
	ydistCalc =mrpResult[1, :] / mrpResult[2, :]
	kxeYe = np.dot(Mi, mrpResult)
	Xe = kxeYe[0, :] / kxeYe[2, :]
	Ye = kxeYe[1, :] / kxeYe[2, :]

	#distorted pixel calculations
	distPixelX  = Xe + (Xe-imageCenter[0])*(k1 * (xdistCalc**2 + ydistCalc**2))
	distPixelY  = Ye + (Ye-imageCenter[1])*(k1 * (xdistCalc**2 + ydistCalc**2))

	# error calculation
	ex = (data[:, 0] - distPixelX)
	ey = (data[:, 1] - distPixelY)

	# error magnitude
	error = np.sqrt(ex**2 + ey**2)

	#print(error) uncomment to see error changing withiteration
	return error

"""
This function  peforms the nonlinear optimization. It constructs a parameter object that holds the
parameters to be optimised. For further information see LMFIT http://cars9.uchicago.edu/software/python/lmfit/
Input:
focalLength: Focal length
translationVector: Translation vector
R: Rotation matrix
k1: first radial distortion coefficient
data: data to compare model against. in this case it pixel locations of the calibration points
sx: Horizontal scaling factor
xyz: world coordinates for calibration points
imageCenter: Center of image [x,y]
pixelDimensions: pixel size [size of x,size of y]
"""

def optimiseCalibration(focalLength, translationVector, R, k1, data, sx, xyz, imageCenter, pixelDimensions):
	params = Parameters()
	params.add('ty', value=translationVector[1].item()) #,vary=False) to exclude value from optimization
	params.add('tx', value=translationVector[0].item())
	params.add('tz', value=translationVector[2].item())
	params.add('focalLength', value=focalLength)
	params.add('k1', value=0)
	params.add('sx', value=sx,vary=False)

	# minimizes our objective function (residual) using the input paramters
	out = minimize(residual, params, method="leastsq",args=(data,R,xyz,imageCenter,pixelDimensions))
	endError = out.residual
	print("\n New Mean Error")
	print(np.mean(endError))
	print("\n New Standard Deviation")
	print(np.std(endError))
	print("\nend of cam")
	return [out.params, endError]
