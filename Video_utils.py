import imageio
import cv2
from skimage import io
import os
import numpy as np



def extract_video_frames(video_path, extract_dir, save_each = 10):
    
    assert (type(save_each)==int), f"'save_each' must be int, but got {type(save_each)}"
    assert (save_each>=1), f"'save_each' must be >= 1, but got {save_each}"

    if(os.path.isdir(extract_dir)):
        if(len(os.listdir(extract_dir))>0):
            raise ValueError("directory {extract_dir} already exists and is not empty")
    elif(os.path.isdir(extract_dir)==False):
        os.mkdir(extract_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        retval,frame = cap.read()
        if(frame is None):
            cap.release()
            break
        elif((count%save_each) == 0):
            #frame = cv2.rotate(frame, cv2.ROTATE_180) # sometimes not needed, seems to be a bug of cv2
            cv2.imwrite(os.path.join(extract_dir,f'frame_{(count//save_each):03d}.jpg'), frame)
        count+=1

def compress_images(image_dir, compressed_dir, factor):

    assert ((factor>=0.1) & (factor <= 1)), f"not valid compressing factor: {factor}, must be in [0.1,1]"
    
    image_paths=[os.path.join(image_dir,name) for name in os.listdir(image_dir)]
    if(os.path.isdir(compressed_dir)):
        if(len(os.listdir(compressed_dir))>0):
            raise ValueError("directory {compressed_dir} already exists and is not empty")
    elif(os.path.isdir(compressed_dir)==False):
        os.mkdir(compressed_dir)
    paths_to_save = [os.path.join(compressed_dir,name) for name in os.listdir(image_dir)]
    
    for image_path, path_to_save in zip(image_paths, paths_to_save):
        image = io.imread(image_path)
        h,w = (np.sqrt(factor) * np.array(image.shape[:2])).astype(int)
        image_compressed = cv2.resize(image,(w,h))
        io.imsave(path_to_save, image_compressed)

def remove_frames_dir(frame_dir):
    
    f_names = os.listdir(frame_dir)
    for f_name in f_names:
        os.remove(os.path.join(frame_dir, f_name))
    os.rmdir(frame_dir)



### stick frames together to make a video
def make_video(frame_dir, video_name, fps=20):
    
    frame_paths=[os.path.join(frame_dir,name) for name in os.listdir(frame_dir)]
    frame_paths.sort()
    
    image = cv2.imread(frame_paths[0])
    height,width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=fps, frameSize=(width,height))
    video.write(image)
    
    for frame_path in frame_paths[1:]:
        image = cv2.imread(frame_path)
        video.write(image)
    
    frame_paths.sort(reverse=True)
    for frame_path in frame_paths[1:]:
        image = cv2.imread(frame_path)
        video.write(image)

    #remove_frames_dir(frame_dir)
    
### stick frames together to make a gif
def make_gif(frame_dir, gif_name, fps=20):
    
    frame_paths=[os.path.join(frame_dir,name) for name in os.listdir(frame_dir)]
    frame_paths.sort()
    
    frames=[]
    
    for frame_path in frame_paths:
        image = io.imread(frame_path)
        frames.append(image)
    
    frame_paths.sort(reverse=True)
    for frame_path in frame_paths[1:]:
        image = io.imread(frame_path)
        frames.append(image)
    
    imageio.mimsave(gif_name, frames, 'GIF', duration=1./fps)
    
    #remove_frames_dir(frame_dir)
    
