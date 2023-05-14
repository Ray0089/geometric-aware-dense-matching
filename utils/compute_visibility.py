
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils.ply import load_ply
import matplotlib.pyplot as plt
import matplotlib.patches as patches
'''
Function used to Import csv Points
'''
def importPoints(fileName, dim):

	p = np.empty(shape = [0,dim]) # Initialize points p
	for line in open(fileName): #For reading lines from a file, loop over the file object. Memory efficient, fast, and simple:
		line = line.strip('\n') # Get rid of tailing \n
		line = line.strip('\r') # Get rid of tailing \r
		x,y,z = line.split(",") # In String Format
		p = np.append(p, [[float(x),float(y),float(z)]], axis = 0) 

	return p

'''
Function used to Perform Spherical Flip on the Original Point Cloud
'''
def sphericalFlip(points, center, param):

	n = len(points) # total n points
	points = points - np.repeat(center, n, axis = 0) # Move C to the origin
	normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
	R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere
	
	flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
	flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
	flippedPoints += points 

	return flippedPoints

'''
Function used to Obtain the Convex hull
'''
def convexHull(points):

	points = np.append(points, [[0,0,0]], axis = 0) # All points plus origin
	hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.

	return hull


'''
Main Function:
Apply Hidden Points Removal Operator to the Given Point Cloud
'''
def Main():
    mesh = load_ply("/home/ray/codes/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/ply_new/002_master_chef_can/poisson.ply")

    myPoints = mesh['pts']

    C = np.array([[0.5,0,0.5]]) # View Point, which is well above the point cloud in z direction
    flippedPoints = sphericalFlip(myPoints, C, math.pi) # Reflect the point cloud about a sphere centered at C
    myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud


    fig = plt.figure(figsize = plt.figaspect(0.5))
    plt.title('Cloud Points With All Points (Left) vs. Visible Points Viewed from Well Above (Right)')

	# First subplot
    ax = fig.add_subplot(1,2,1, projection = '3d')
    ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

	# Second subplot
    ax = fig.add_subplot(1,2,2, projection = '3d')
    for vertex in myHull.vertices[:-1]: # Exclude Origin, which is the last element
	    ax.scatter(myPoints[vertex, 0], myPoints[vertex, 1], myPoints[vertex, 2], c='b', marker='o') # Plot visible points
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    fig.savefig('test_plt.png', dpi=600)
    # plt.show()
	# plt.savefig()

    return 

'''
Test Case:
A Sphere ranged from -1 to 1 in three axes with 961 Points is used for testing.
Viewed from well above: (0,0,10)
'''
def Test():
	myPoints = importPoints('sphere.csv', 3) # Import the Test Point Cloud

	C = np.array([[0,0,10]]) # 10 is well above the peak of circle which is 1
	flippedPoints = sphericalFlip(myPoints, C, math.pi)
	myHull = convexHull(flippedPoints)

	# Plot
	fig = plt.figure(figsize = plt.figaspect(0.5))
	plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')

	# First subplot
	ax = fig.add_subplot(1,2,1, projection = '3d')
	ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	# Second subplot
	ax = fig.add_subplot(1,2,2, projection = '3d')
	for vertex in myHull.vertices[:-1]:
		ax.scatter(myPoints[vertex, 0], myPoints[vertex, 1], myPoints[vertex, 2], c='b', marker='o') # Plot visible points
	ax.set_zlim3d(-1.5, 1.5)
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()

	return 

'''
Onsite Coding Challenge
@author Hai Tang (haitang@jhu.edu)
Nov 12, 2015.
'''
def VisiblePoints(pts,cam_center):
	flag = np.zeros(len(pts), int) # Initialize the points visible from possible 6 locations. 0 - Invisible; 1 - Visible.
	
	flippedPoints = sphericalFlip(pts, cam_center, math.pi)
	myHull = convexHull(flippedPoints)
	visibleVertex = myHull.vertices[:-1] # indexes of visible points
	return visibleVertex


def InvisiblePoints():
	myPoints = importPoints('points.csv', 3) # Import the Given Point Cloud

	############################ Method 1: Use a flag array indicating visibility, most efficent in speed and memory ############################
	flag = np.zeros(len(myPoints), int) # Initialize the points visible from possible 6 locations. 0 - Invisible; 1 - Visible.
	C = np.array([[[0,0,100]], [[0,0,-100]], [[0,100,0]], [[0,-100,0]], [[100,0,0]], [[-100,0,0]]])  # List of Centers
	for c in C:
		flippedPoints = sphericalFlip(myPoints, c, math.pi)
		myHull = convexHull(flippedPoints)
		visibleVertex = myHull.vertices[:-1] # indexes of visible points
		flag[visibleVertex] = 1
	invisibleId = np.where(flag == 0)[0] # indexes of the invisible points


	# Plot for method 1
	fig = plt.figure(figsize = plt.figaspect(0.5))
	plt.title('Cloud Points With All Points (Left) vs. Invisible Points (Right)')
	
	# First subplot
	ax = fig.add_subplot(1,2,1, projection = '3d')
	ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	# Second subplot
	ax = fig.add_subplot(1,2,2, projection = '3d')
	for i in invisibleId:
		ax.scatter(myPoints[i, 0], myPoints[i, 1], myPoints[i, 2], c='b', marker='o') # Plot visible points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()

def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def visualize(img,K,corner_3d,gt_rt,pred_rt):
	
	corner_2d_gt = project(corner_3d, K, pose_gt)
	corner_2d_pred = project(corner_3d, K, pose_pred)

	_, ax = plt.subplots(1)
	ax.imshow(img)
	ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
	ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
	ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
	ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
	plt.show()

	return 

def visualize(img,K,corner_3d,gt_rt,pred_rt):

	#transformation = np.array([[1,0,0,0.000216],[0,1,0,-0.002053],[0,0,1,-0.013238],[0,0,0,1]])

	#gt_rt = np.matmul(gt_rt ,transformation.T)
	
	corner_2d_gt = project(corner_3d, K, gt_rt)
	#corner_2d_pred = project(corner_3d, K, pose_pred)

	_, ax = plt.subplots(1)
	ax.imshow(img)
	# for pt in corner_2d_gt:
	# 	ax.scatter(pt[0],pt[1],s=1,c='red')
	ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
	ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
	ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
	ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
	plt.savefig('test.jpg',dpi=600)


if __name__ == '__main__':
	Main()


