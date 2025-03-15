import pulse2percept as p2p
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import cv2

import sys
sys.path.append('../ZoeDepth')
from zoedepth.utils.misc import get_image_from_url, colorize
from PIL import Image

def compute_right_eye(left_img, depth_map, fov=60, shift=0.063):
    fov_rad = fov * math.pi / 180
    width = depth_map.shape[1]

    x_map, y_map = np.meshgrid(np.arange(depth_map.shape[1]), 
                               np.arange(depth_map.shape[0]))

    x_shifted = x_map + width*shift/(2*depth_map*np.tan(fov_rad/2))
    x_shifted = np.clip(x_shifted, 0, width-1)
    right_img = cv2.remap(left_img, x_shifted.astype(np.float32), 
                          y_map.astype(np.float32), cv2.INTER_LINEAR)
    
    return right_img

# start with a really idealistic prosthetic with tons of electrodes
def get_big_prosthesis(eye):
    earray = p2p.implants.ElectrodeArray([])
    e_count = 0
    for i in range(-3000, 3000, 100):
        for j in range(-3000, 3000, 100):
            earray.add_electrode(e_count, p2p.implants.DiskElectrode(i, j, z=0, r=25))
            e_count += 1
    return p2p.implants.ProsthesisSystem(earray, eye=eye)

def get_percept_data_from_image(image, model, prosthesis):
    prosthesis.stim = p2p.stimuli.ImageStimulus(image)
    percept = model.predict_percept(prosthesis)
    return percept.data

def get_keypoint_positions(img_np, depth_map, num_to_include, fov=60, shift=0.063):
    image_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image_cv2, None)

    img_keypoints = cv2.drawKeypoints(img_np, keypoints, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    kp_positions_left = [kp.pt for kp in keypoints][:num_to_include]
    
    width = img_np.shape[1]
    fov_rad = fov * math.pi / 180
    kp_positions_right = [(j - width*shift/(2*depth_map[int(i)][int(j)]*np.tan(fov_rad/2)), i) 
                          for (j, i) in kp_positions_left]
    
    return kp_positions_left, kp_positions_right, img_keypoints

def get_keypoint_image(img_shape, kp_positions, kp_radius=10):
    kp_img = np.zeros(img_shape)
    for (j, i) in kp_positions:
        cv2.circle(kp_img, (int(j), int(i)), radius=kp_radius, 
                   color=(255, 255, 255), thickness=-1)
    return kp_img

# we can normalize the electrode positions into the image dimensions. 
# then for each key point, find the closest normalized electrode. 
# then mark that electrode as illuminated
# then use that as the stim
def normalize(x, y, x_min, x_max, y_min, y_max, x_min_new, x_max_new, y_min_new, y_max_new):
    x_new = x_max_new - (x_max_new - x_min_new)/(x_max - x_min)*(x_max - x)
    y_new = y_max_new - (y_max_new - y_min_new)/(y_max - y_min)*(y_max - y)
    return (x_new, y_new)

# L2 dist
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_stim_from_key_points(implant, keypoint_positions, img_shape):
    # this is how the Scoreboard model's plot gets its Cartesian size from xrange and yrange
    # i think the image is scaled to fit the topography map. 
    wm = p2p.topography.Watson2014Map()

    # normalize
    max_orig, min_orig = wm.dva_to_ret(12, 12) # numbers from xrange/yrange vals
    min_x_new = 0
    min_y_new = 0
    (max_y_new, max_x_new) = img_shape
    normalized_electrode_positions = []
    for name in implant.electrode_names:
        normalized_electrode_positions.append(normalize(
                                                implant.electrodes[name].x,
                                                implant.electrodes[name].y,
                                                min_orig, max_orig, 
                                                min_orig, max_orig, 
                                                min_x_new, max_x_new, 
                                                min_y_new, max_y_new))

    # go through keypoints to find the closest:
    nearest_electrodes = []
    for kpos in keypoint_positions:
        min_dist = 1e9
        min_electrode = -1
        for eind, epos in enumerate(normalized_electrode_positions):
            d = dist(kpos, epos)
            if d < min_dist:
                min_dist = d
                min_electrode = eind
        nearest_electrodes.append(min_electrode)

    # mark electrode as illuminated
    stim = [0] * implant.n_electrodes
    for electrode in nearest_electrodes:
        stim[electrode] = 1
    return stim

