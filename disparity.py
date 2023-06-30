# import OpenCV and pyplot 
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import open3d

# read left and right images
#imgRR = cv.imread('onemRu.png')
imgR = cv.imread('curtainru.png', cv.IMREAD_GRAYSCALE)
imgL = cv.imread('curtainlu.png', cv.IMREAD_GRAYSCALE)

cropped_imageL = imgL#[60:900, 230:1500]
cropped_imageR = imgR#[60:900, 230:1500]
 
# Display cropped image
#cv.imshow("cropped", cropped_image)

# creates StereoBm object 
stereo = cv.StereoSGBM_create(numDisparities =160,
                            blockSize =1)
  
# computes disparity
disparity = stereo.compute(cropped_imageL, cropped_imageR)
#Hori = np.concatenate((imgRR, disparity), axis=1)
# displays image as grayscale and plotted
#combined_image = np.hstack((imgL, imgR, disparity))
#plt.imshow(imgRR)
print(disparity)
plt.imshow(disparity)
arr = np.array(disparity)
print(arr.shape)
plt.show()


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''




def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

print('generating 3d point cloud...',)
h, w = imgL.shape[:2]
f = 0.8*w                         # guess for focal length

Q=np.float32(
[[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.67034456e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.03445789e+03],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.16584610e+02],
 [ 0.00000000e+00,  0.00000000e+00, -8.44895915e-04,  0.00000000e+00]])


#Q = np.float32([[1, 0, 0, -0.5*w],
#                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#                    [0, 0, 0,     -f], # so that y-axis looks up
#                   [0, 0, 1,      0]])


disp = disp.astype(np.float32) * 1.0/255
points = cv.reprojectImageTo3D(disp, Q, True)

#reflect on x axis
reflect_matrix = np.identity(3)
reflect_matrix[0] *= -1
points = np.matmul(points,reflect_matrix)

#extract colors from image
colors = cv.cvtColor(cv.imread('curtainlu.png'), cv.COLOR_BGR2RGB)

#filter by min disparity
img=disp.copy()
mask = img > img.min()
out_points = points[mask]
out_colors = colors[mask]

#filter by dimension
idx = np.fabs(out_points[:,0]) < 4.5
out_points = out_points[idx]
out_colors = out_colors.reshape(-1, 3)
out_colors = out_colors[idx]

write_ply('out.ply', out_points, out_colors)
print('%s saved' % 'out.ply')

"""
points = np.reshape(points, (-1, 3))
colors = np.reshape(cv.imread('curtainlu.png'), (-1, 3)).astype(np.float32) / 255.0
colors = colors[..., ::-1]
points_mask = np.logical_and(points[..., 2] > -100000, points[..., 2] < -2600)
colors = colors[points_mask]
points = points[points_mask]

out_fn = 'out.ply'
write_ply(out_fn, points, colors)
print('%s saved' % out_fn)


pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(points * np.array((1, 1, -1)))
pcd.colors = open3d.Vector3dVector(colors)
open3d.draw_geometries([pcd])
"""
#cv.imshow('left', imgL)
#cv.imshow('disparity', (disp-min_disp)/num_disp)
#cv.waitKey()

print('Done')



# Setting parameters for StereoSGBM algorithm
"""
numDisparities = 180;
blockSize =3;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;

def nothing(x):
    pass

 
cv.namedWindow('disp',cv.WINDOW_NORMAL)
cv.resizeWindow('disp',600,600)

cv.createTrackbar('numDisparities','disp',1,1600,nothing)
cv.createTrackbar('blockSize','disp',5,50,nothing)
cv.createTrackbar('preFilterType','disp',1,1,nothing)
cv.createTrackbar('preFilterSize','disp',2,25,nothing)
cv.createTrackbar('preFilterCap','disp',5,62,nothing)
cv.createTrackbar('textureThreshold','disp',10,100,nothing)
cv.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv.createTrackbar('speckleRange','disp',0,100,nothing)
cv.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv.createTrackbar('minDisparity','disp',5,25,nothing)

stereo = cv.StereoSGBM_create()

while True:
    numDisparities = cv.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv.getTrackbarPos('blockSize','disp')*2 + 5
  #  preFilterType = cv.getTrackbarPos('preFilterType','disp')
  #  preFilterSize = cv.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv.getTrackbarPos('preFilterCap','disp')
 #   textureThreshold = cv.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv.getTrackbarPos('minDisparity','disp')

    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
  #  stereo.setPreFilterType(preFilterType)
  #  stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
  #  stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    disp = stereo.compute(imgL, imgR).astype(np.float32)
    disp = (disp/16.0 - minDisparity)/numDisparities
    cv.imshow("disp",disp)

    if cv.waitKey(1) == 27:
      break
"""
# Creating an object of StereoSGBM algorithm

'''
stereo = cv.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange

    )
 
# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv.normalize(disp,0,255,cv.NORM_MINMAX)
 
# Displaying the disparity map
cv.imshow("disparity",disp)
cv.waitKey(0)
'''