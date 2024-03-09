import cv2
from skimage import io
from skimage.exposure import match_histograms
import os
import numpy as np
from Video_utils import remove_frames_dir



### check if points lie in given bounds of image
def is_in_bounds(points, bounds):
    x1,y1=bounds[0]
    x2,y2=bounds[1]
    return [1 if ((p[0]>=x1) & (p[0]<=x2) & (p[1]>=y1) & (p[1]<=y2))
           else 0 for p in points]

def extract_key_points(image, **kwargs):
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(image, None)
    
    if('bounds' in kwargs.keys()):
        bounds = kwargs.pop('bounds')
        mask_is_in_bounds = is_in_bounds([p.pt for p in kp], bounds)
        kp = tuple([kp[i] for i in range(len(kp)) if mask_is_in_bounds[i]==1])
        des = np.array([des[i] for i in range(len(des)) if mask_is_in_bounds[i]==1])
        
    return kp, des

def match_key_points(des1, des2):
    #create BFMatcher object
    bf = cv2.BFMatcher()
    #Match descriptors
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    return sorted(matches, key = lambda x:x.distance)

### select best matches with RANSAC
def findHomography(matches, base_kp, kp):
    kp1,kp2 = base_kp,kp
    src_pts = np.array([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    inliers = [matches[i] for i in range(len(matches)) if mask[i]==1]
    return H, inliers

### find shift to put geometric center of key points in the center of the image
def shift_to_center_by_matches(image, matches, kp):
    
    h,w,_ = image.shape
    x_mean, y_mean = (np.mean([kp[m].pt[0] for m in matches]),
                              np.mean([kp[m].pt[1] for m in matches]))
    
    return np.array([w/2 - x_mean, h/2 - y_mean])

### find shift to allign geometric centers of two sets of key points
def shift_by_matches(matches, base_kp, kp):
    kp1,kp2 = base_kp, kp
    
    x_center_1, y_center_1 = (np.mean([kp1[m.queryIdx].pt[0] for m in matches]),
                              np.mean([kp1[m.queryIdx].pt[1] for m in matches]))
    
    x_center_2, y_center_2 = (np.mean([kp2[m.trainIdx].pt[0] for m in matches]),
                              np.mean([kp2[m.trainIdx].pt[1] for m in matches]))
    
    return np.array([x_center_1-x_center_2, y_center_1-y_center_2])

### find zoom coefficient between two sets of key points 
def zoom_by_matches(matches, base_kp, kp):
    
    kp1,kp2=base_kp,kp
    
    # find zoom coefficient by std of points cloud
    # x_1 = np.array([kp1[m.queryIdx].pt[0] for m in matches])
    # x_2 = np.array([kp2[m.trainIdx].pt[0] for m in matches])
    # y_1 = np.array([kp1[m.queryIdx].pt[1] for m in matches])
    # y_2 = np.array([kp2[m.trainIdx].pt[1] for m in matches])
    # c_x = np.std(x_1)/np.std(x_2)
    # c_y = np.std(y_1)/np.std(y_2)
    # c_x = c_y = np.sqrt(c_x*c_y)
    
    # find zoom coefficient by distance to mass center of points
    center_1 = np.mean([kp1[m.queryIdx].pt for m in matches], axis=0)
    center_2 = np.mean([kp2[m.trainIdx].pt for m in matches], axis=0)
    c_x = c_y = np.mean(
                [np.linalg.norm(kp1[m.queryIdx].pt-center_1)/np.linalg.norm(kp2[m.trainIdx].pt-center_2)
                for m in matches])
    
    assert ((c_x > 0.9) & (c_y > 0.9)), f"failed to find zoom coefficient"
    return np.array([c_x, c_y])

### find rotation angle to allign orientation of two sets of key points
def rotation_by_matches(matches, base_kp, kp):
    
    kp1,kp2=base_kp,kp
    
    center_1 = np.mean([kp1[m.queryIdx].pt for m in matches], axis=0)
    center_2 = np.mean([kp2[m.trainIdx].pt for m in matches], axis=0)
    
    P1 = np.array([kp1[m.queryIdx].pt - center_1 for m in matches]).transpose()
    P2 = np.array([kp2[m.trainIdx].pt - center_2 for m in matches]).transpose()
    
    S = P1 @ P2.transpose()
    
    U, Sigma, Vh = np.linalg.svd(S, full_matrices=False) #SVD decomposition S = U*Sigma*Vh 
    if(np.linalg.det(Vh.transpose() @ U.transpose()) < 0):
        U[:,-1] *= -1
    R = Vh.transpose() @ U.transpose()
    
    angle = np.arctan(R[1,0]/R[0,0]) #R = [[cos a, -sin a],
                                      #    [sin a , cos a]]
    return angle

def shift_image(image, shift, **kwargs):
    
    shift_x,shift_y=shift
    
    translation_matrix = np.float32([[1,0,shift_x], 
                                     [0,1,shift_y]])   
    image_shifted = cv2.warpAffine(image, translation_matrix, image.shape[1::-1])
    
    if('kp' in kwargs.keys()):
        kp = kwargs.pop('kp')
        for p in kp:
            p.pt=p.pt+shift

    return image_shifted

### Zoom image by a given zoom coefficient 
### and extract central part of it so that image size doesn't change.
def zoom_image(image, zoom_coefficient, shape, **kwargs):
    w, h = shape
    c_x, c_y = zoom_coefficient
    
    interpolation =  cv2.INTER_LINEAR 
    image_zoomed = cv2.resize(image,(int(c_x*w),int(c_y*h)),interpolation=interpolation)
    image_zoomed = image_zoomed[int(h*(c_y-1)/2):int(h*(c_y+1)/2),int(w*(c_x-1)/2):int(w*(c_x+1)/2),:]
    
    if('kp' in kwargs.keys()):
        kp = kwargs.pop('kp')
        for p in kp:
            p.pt=(p.pt[0]*c_x - (c_x-1)*w/2, p.pt[1]*c_y - (c_y-1)*h/2) 
    
    return image_zoomed

### Rotate image around 'center' by 'angle' in radians.
### Result image has the same size
def rotate_image(image, angle, center, **kwargs):
    
    angle = angle/np.pi*180
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)  
    image_rotated = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1]) 
    
    if('kp' in kwargs.keys()):
        kp = kwargs.pop('kp')
        for p in kp:
            p.pt=rotation_matrix @ np.array([p.pt[0],p.pt[1],1])

    return image_rotated



### 'image2' is zoomed, then shifted, then rotated 
### in a way that key points of 'image1' lying in 'bounds'
### are alligned with corresponding key points of 'image2'  
def make_zoom_shift_rotation(image1, image2, bounds, **kwargs):
    
    kp1, des1 = extract_key_points(image1, bounds=bounds)
    kp2, des2 = extract_key_points(image2)
    matches = match_key_points(des1, des2)
    H, matches = findHomography(matches, kp1, kp2)
    
    zoom = zoom_by_matches(matches,kp1,kp2)
    image2_zoomed = zoom_image(image2,zoom,image1.shape[1::-1],kp=kp2)

    shift = shift_by_matches(matches,kp1,kp2)
    image2_zoomed_shifted = shift_image(image2_zoomed,shift,kp=kp2)
    if('shifts' in kwargs.keys()):
        shifts = kwargs.pop('shifts')
        shifts.append(shift)
    
    angle = rotation_by_matches(matches,kp1,kp2)
    image2_zoomed_shifted_rotated = rotate_image(
        image2_zoomed_shifted,angle,np.array(image1.shape[1::-1])/2)
    if('angles' in kwargs.keys()):
        angles = kwargs.pop('angles')
        angles.append(angle)
    
    return image2_zoomed_shifted_rotated

def make_homography(image1, image2, bounds):
    
    kp1, des1 = extract_key_points(image1, bounds=bounds)
    kp2, des2 = extract_key_points(image2)
    matches = match_key_points(des1, des2)
    H, matches = findHomography(matches, kp1, kp2)

    h,w = image1.shape[:2]
    I = np.zeros((h,w,3))
    image2_projected = np.uint8(I)
    cv2.warpPerspective(src=image2, dst=image2_projected, M=H, dsize=(w,h), 
                        flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP, 
                        borderMode=cv2.BORDER_TRANSPARENT) 

    return image2_projected



### crop empty area of frames due to shifts and rotations
def crop_frames(frame_paths,shifts,angles):
    
    image_0 = io.imread(frame_paths[0])
    H,W = image_0.shape[:2]
    
    # empty area due to shifts 
    shifts.append(np.array([0,0]))
    x_l=max([int(shift[0]) for shift in shifts if shift[0]>=0])
    x_r=-min([int(shift[0]) for shift in shifts if shift[0]<=0])
    y_u=max([int(shift[1]) for shift in shifts if shift[1]>=0])
    y_d=-min([int(shift[1]) for shift in shifts if shift[1]<=0])
    
    # empty area due to rotations
    angles.append(0)
    a = np.max(np.abs(angles))
    if(a==0):
        d_x=d_y=0
    else:
        d_x = int((H/2 - W*np.sqrt((1-np.cos(a))/(2*np.sin(a)**2)-1/4))*np.tan(a))
        d_y = int((W/2 - H*np.sqrt((1-np.cos(a))/(2*np.sin(a)**2)-1/4))*np.tan(a))
    
    # cropped image bounds
    L = int(x_l/np.cos(a) + d_x)
    U = int(y_u/np.cos(a) + d_y)
    R = W - int((x_r/np.cos(a) + d_x))
    D = H - int((y_d/np.cos(a) + d_y))
    
    for frame_path in frame_paths:
        image = io.imread(frame_path)
        image_crop = image[U:D,L:R,:]
        io.imsave(frame_path, image_crop)

### allign color histograms of frames with that of the first frame
def correct_frames_gamma(frame_paths):
    
    reference_image = io.imread(frame_paths[0])
    for frame_path in frame_paths[1:]:
        image = io.imread(frame_path)
        image_gamma = match_histograms(image,reference_image,channel_axis=2)
        io.imsave(frame_path, image_gamma)

### Zoom given photos taken from different distcances
### in a way that the object in 'base_object_bounds' keeps the same size.
### Photos are supposed to be sorted by distance they were taken from.
### Watching obtained series of zoomed photos will give the dolly-zoom effect. 
def make_frames(images, base_object_bounds, frame_dir, correct_gamma=False):
    
    if(os.path.isdir(frame_dir)):
        if(len(os.listdir(frame_dir))>0):
            raise ValueError("directory {frame_dir} already exists and is not empty")
    elif(os.path.isdir(frame_dir)==False):
        os.mkdir(frame_dir)
    
    bounds=np.copy(base_object_bounds)
    shifts=[]
    angles=[]
    
    # adjust the first frame, geometric center of key points is shifted to the image center
    image1 = io.imread(images[0])
    image2 = io.imread(images[1])
    kp1, des1 = extract_key_points(image1, bounds=bounds)
    kp2, des2 = extract_key_points(image2)
    matches = match_key_points(des1, des2)
    H, matches = findHomography(matches, kp1, kp2)
    shift = shift_to_center_by_matches(image1, [m.queryIdx for m in matches], kp1)
    image_adjusted = shift_image(image1, shift)
    bounds = bounds + shift
    shifts.append(shift)
    io.imsave(os.path.join(frame_dir,"frame_000.jpg"),image_adjusted)
    
    # remaining frames are adjusted in accordance with the first frame
    for i in range(1,len(images)):
        image = io.imread(images[i])
        image_adjusted = make_zoom_shift_rotation(image_adjusted, image, bounds, 
                                                  shifts=shifts, angles=angles)
        
        # instead of our zoom-shift-rotate approach one can just calculate homography matrix 
        # and use it for transformation, nevertheless in practice this approche changes 
        # dramatically the perspective and resulting video have little visual relevance
        # with dolly-zoom effect
        # image_adjusted = make_homography(image_adjusted, image, bounds)
        
        io.imsave(os.path.join(frame_dir,f"frame_{i:03d}.jpg"),image_adjusted)
    
    frame_paths=[os.path.join(frame_dir,name) for name in os.listdir(frame_dir)]
    frame_paths.sort()
    crop_frames(frame_paths, shifts, angles)
    if(correct_gamma):
        correct_frames_gamma(frame_paths)