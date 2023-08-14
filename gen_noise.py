# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""

import numpy as np
#import tensorflow as tf 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os 
import shutil
import random
import math
import scipy.io as sio
import time
#from skimage import measure
#import binvox_rw
import argparse
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.utils.libkdtree import KDTree
import re
from approxmatch import tf_approxmatch
import sys

args = sys.argv


def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)

def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)

def distance_matrix(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances   

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    #distances = tf.reduce_mean(distances)
    return distances

def vis_single_points(points, plyname): 
    
    
    header = "ply\n" \
             "format ascii 1.0\n" \
             "element vertex {}\n" \
             "property double x\n" \
             "property double y\n" \
             "property double z\n" \
             "property uchar red\n" \
             "property uchar green\n" \
             "property uchar blue\n" \
             "end_header\n".format(points.shape[0])
    with open(plyname, 'w') as f:
        f.write(header)
        for i in range(int(points.shape[0])):
            f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2], 255, 0, 0))

def noise_points_produce(data_path):
    mesh = trimesh.load(data_path, force='mesh')
    pointclouds, _= mesh.sample(50000, return_index=True)
    pointclouds = np.asarray(pointclouds).reshape(-1,3)
    noises = []
    for _ in range(10):
        noise = np.random.randn(pointclouds.shape[0],pointclouds.shape[1])
        noise = pointclouds + noise * 0.01
        noises.append(noise)
    noises = np.asarray(noises).reshape(-1,3)
    vis_single_points(noises,"./noise_chair.ply")
    # pnts = noise
    # ptree = cKDTree(pnts)
    # sigmas = []
    # for p in np.array_split(pnts,100,axis=0):
    #     d = ptree.query(p,51)
    #     sigmas.append(d[0][:,-1])
    
    # if(i%5==0):
    #     sigmas = np.concatenate(sigmas)
    #     #print(np.max(sigmas),np.min(sigmas),np.mean(sigmas))
    #     tt = pnts + 1.0*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
    #     sample.append(tt)

def process_data(data_dir, dataname):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = np.load(os.path.join(data_dir, dataname)) + '.xyz'
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.')
        exit()
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale


noise_points_produce(data_path=args[1])
        
    
                    
    
    
