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
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--dataset', type=str, default="other")
parser.add_argument('--dataname', type=str, default="other")
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx


#GT_DIR = '/data/mabaorui/common_data/ShapeNetCore.v1/' + a.class_idx + '/'
GT_DIR = '/data/mabaorui/DFAUST/gt/'

BS = 1
POINT_NUM = 5000
INPUT_DIR = a.data_dir
#INPUT_DIR = '/home/mabaorui/AtlasNetOwn/data/sphere/'
OUTPUT_DIR = a.out_dir

TRAIN = a.train
bd = 0.55

if(TRAIN):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)


def vis_single_points_with_color(points, colors, plyname): 
    
    
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
            f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]))

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

#        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
#        # Handle normals that point into wrong direction gracefully
#        # (mostly due to mehtod not caring about this in generation)
#        normals_dot_product = np.abs(normals_dot_product)
        
        normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
        normals_dot_product = normals_dot_product.sum(axis=-1)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty


        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        #print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        print('chamferL2:',chamferL2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('normals_correctness:',normals_correctness,'chamferL1:',chamferL1)
        return normals_correctness, chamferL1, chamferL2

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

# def noise_points_produce():
#     mesh = trimesh.load("./chair.off", force='mesh')
#     pointclouds, _= mesh.sample(50000, return_index=True)
#     noise = np.random.randn(pointclouds.shape[0]*10,pointclouds.shape[1])
#     noise = pointclouds + noise * 0.01
#     vis_single_points(noise,"./noise_chair.ply")
#     # pnts = noise
#     # ptree = cKDTree(pnts)
#     # sigmas = []
#     # for p in np.array_split(pnts,100,axis=0):
#     #     d = ptree.query(p,51)
#     #     sigmas.append(d[0][:,-1])
    
#     # if(i%5==0):
#     #     sigmas = np.concatenate(sigmas)
#     #     #print(np.max(sigmas),np.min(sigmas),np.mean(sigmas))
#     #     tt = pnts + 1.0*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
#     #     sample.append(tt)

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
    
    POINT_NUM_GT = pointcloud.shape[0]
    QUERY_EACH = 2000000//POINT_NUM_GT

    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
    
    sigmas = np.concatenate(sigmas)
    sample = []

    for i in range(QUERY_EACH):
        scale = 0.05 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1,POINT_NUM,3)

        
    sample = np.asarray(sample).reshape(-1,3)

    np.savez(os.path.join(data_dir, dataname)+'.npz', sample = sample, noise_point = pointcloud, trans = shape_center, scal = shape_scale)



process_data(a.data_dir, a.dataname)
SHAPE_NUM = 2


feature = tf.placeholder(tf.float32, shape=[None,SHAPE_NUM])
input_points_3d = tf.placeholder(tf.float32, shape=[None,3])

# feature = tf.placeholder(tf.float32, shape=[POINT_NUM,SHAPE_NUM])
# input_points_3d = tf.placeholder(tf.float32, shape=[POINT_NUM,3])

points_target = tf.placeholder(tf.float32, shape=[BS,POINT_NUM,3])




def local_decoder(feature_f,input_points_3d_f):
    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
        
        feature_f = tf.nn.relu(tf.layers.dense(feature_f,128))
        net = tf.nn.relu(tf.layers.dense(input_points_3d_f, 512))
        net = tf.concat([net,feature_f],1)

        with tf.variable_scope('dis_decoder', reuse=tf.AUTO_REUSE):
            for i in range(8):
                with tf.variable_scope("resnetBlockFC_%d" % i ):
                    b_initializer=tf.constant_initializer(0.0)
                    w_initializer = tf.random_normal_initializer(mean=0.0,stddev=np.sqrt(2) / np.sqrt(512))
                    #net = tf.layers.dense(tf.nn.relu(net),512,kernel_initializer=w_initializer,bias_initializer=b_initializer)
                    net = tf.layers.dense(tf.nn.relu(net),512)
                    
        b_initializer=tf.constant_initializer(-0.5)
        w_initializer = tf.random_normal_initializer(mean=2*np.sqrt(np.pi) / np.sqrt(512), stddev = 0.000001)
        print('net:',net)
        sdf = tf.layers.dense(tf.nn.relu(net),1,kernel_initializer=w_initializer,bias_initializer=b_initializer)
        #sdf = tf.layers.dense(tf.nn.relu(net),1)
        grad = tf.gradients(ys=sdf, xs=input_points_3d) 
        print('grad',grad)
        print(grad[0])
        normal_p_lenght = tf.expand_dims(safe_norm(grad[0],axis = -1),-1)
        print('normal_p_lenght',normal_p_lenght)
        grad_norm = grad[0]/(normal_p_lenght + 1e-12)
        print('grad_norm',grad_norm)
        
        g_points = input_points_3d - sdf * grad_norm
        return sdf, grad_norm, g_points
    
sdf,grad_norm,points_gen = local_decoder(feature,input_points_3d)

#_,grad_norm_sur,_ = local_decoder(feature,points_gen)

loss_zero, _, _ = local_decoder(feature,points_gen)
loss_zero = tf.reduce_mean(tf.abs(loss_zero))
points_gen = tf.expand_dims(points_gen,0)

match = tf_approxmatch.approx_match(points_target, points_gen)
cost = tf_approxmatch.match_cost(points_target, points_gen, match)
loss_emd = tf.reduce_mean(cost/tf.cast(points_target.shape[1], tf.float32))

points_target_cd = points_target
points_target_cd = tf.reshape(points_target_cd,[-1,3])
points_gen = tf.reshape(points_gen,[-1,3])
print(points_target_cd.shape,points_gen.shape)
dist_input_denoise = av_dist(points_target_cd, points_gen) 
print(dist_input_denoise.shape)
# Eq.6 in the manuscript
loss_geo_consistency = tf.reduce_mean(tf.clip_by_value(tf.abs(sdf) - dist_input_denoise, 0.0, tf.constant(np.inf)))
print('geo loss:',loss_geo_consistency,loss_emd)


loss = loss_emd + 0.1*loss_geo_consistency
# An alternative implementation of the loss function, which is more robust
#loss = loss_emd + 0.1*loss_zero
# points_gen = tf.layers.dense(tf.nn.relu(net),points_target.shape[1]*3)
# points_gen = tf.reshape(points_gen,[BS,POINT_NUM,3])
# print(points_gen)

# match = tf_approxmatch.approx_match(points_target, points_gen)
# cost = tf_approxmatch.match_cost(points_target, points_gen, match)
# loss = tf.reduce_mean(cost/tf.cast(points_target.shape[1], tf.float32))

match_inv = tf_approxmatch.approx_match(points_gen,points_target)
cost_inv = tf_approxmatch.match_cost(points_gen,points_target, match_inv)


t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)


config = tf.ConfigProto(allow_soft_placement=False) 
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)


with tf.Session(config=config) as sess:
    feature_bs = []
    for i in range(SHAPE_NUM):
        tt = []
        for j in range(int(POINT_NUM)):
            t = np.zeros(SHAPE_NUM)
            t[i] = 1
            tt.append(t)
        feature_bs.append(tt)
    feature_bs = np.asarray(feature_bs)
    POINT_NUM_GT_bs = np.array(POINT_NUM).reshape(1,1)
    points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()

        load_data = np.load(os.path.join(a.data_dir, a.dataname)+'.npz')
        point = np.asarray(load_data['noise_point']).reshape(-1,1,3)
        sample = np.asarray(load_data['sample']).reshape(-1,3)
        print(point.shape,sample.shape)
        s_num = sample.shape[0]
        s_num_gt = point.shape[0]
        for i in range(400010):
            
            index_coarse = np.random.choice(100, 1)
            index_fine = np.random.choice(s_num//100, POINT_NUM, replace = False)
            rt = index_fine * 100 + index_coarse
            
            index_coarse = np.random.choice(100, 1)
            index_fine = np.random.choice(s_num_gt//100, POINT_NUM, replace = False)
            rtt = index_fine * 100 + index_coarse
            
            
            input_points_2d_bs = sample[rt].reshape(-1, 3)
            point_gt = point[rtt,0,:].reshape(BS,-1,3)
            
            feature_bs_t = feature_bs[0,:,:].reshape(POINT_NUM,SHAPE_NUM)
            #print("input_points_3d:",input_points_2d_bs.shape,"points_target:",point_gt.shape,"feature:",feature_bs_t.shape)
            _,loss_c,points_gen_c = sess.run([loss_optim,loss,points_gen],feed_dict={input_points_3d:input_points_2d_bs,points_target:point_gt,feature:feature_bs_t})
            if(i%2000 == 0):
                points_gen_c,loss_emd_c, loss_zero_c = sess.run([points_gen, loss_emd, loss_zero],feed_dict={input_points_3d:input_points_2d_bs,points_target:point_gt,feature:feature_bs_t})
                print('epoch:', i, 'epoch loss:', loss_c,'loss_emd_c:', loss_emd_c, 'loss_zero_c:',loss_zero_c)
                
                
                
                # points_gen_c = np.asarray(points_gen_c).reshape(-1,3)
                # # print(points_gen_c)
                # # print(points_gen_c.shape)

                # vis_single_points(points_gen_c, OUTPUT_DIR + 'test_output_' + str(i) + '.ply')
            if(i%10000 == 0):
                print('save model')
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
        # print('save model')
        # saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=j)
        end_time = time.time()
        print('run_time:',end_time-start_time)
    else:
        
        print('test')
        # checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR).all_model_checkpoint_paths
        # print(checkpoint[a.save_idx])
        # saver.restore(sess, checkpoint[a.save_idx])
        
        s = np.arange(-bd,bd, (2*bd)/128)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        POINT_NUM_GT_bs = np.array(vox_size).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        
        POINT_NUM_GT_bs = np.array(vox_size*vox_size).reshape(1,1)

        test_num = SHAPE_NUM
        #test_num = 4
        print('test_num:',test_num)
        cd = 0
        nc = 0
        cd2 = 0
        num = 0
        
        cd1_steps = np.zeros(9)
        nc_steps = np.zeros(9)
        
        sess.run(tf.global_variables_initializer())

        #saver.restore(sess, a.out_dir + 'model-' + str(epoch))
        #print(a.out_dir + 'model-' + str(epoch))
        
        checkpoint = tf.train.get_checkpoint_state(a.out_dir).all_model_checkpoint_paths
        print(checkpoint[a.save_idx])
        saver.restore(sess, checkpoint[a.save_idx])
        
        loc_data = np.load(os.path.join(a.data_dir, a.dataname)+'.npz')
        point_sparse = loc_data['noise_point'].reshape(-1,3)
        
        
        
        input_points_2d_bs = []

        bd_max = [np.max(point_sparse[:,0]), np.max(point_sparse[:,1]), np.max(point_sparse[:,2])] 
        bd_min = [np.min(point_sparse[:,0]), np.min(point_sparse[:,1]),np.min(point_sparse[:,2])] 
        bd_max  = np.asarray(bd_max) + 0.05
        bd_min = np.asarray(bd_min) - 0.05
        sx = np.arange(bd_min[0], bd_max[0], (bd_max[0] - bd_min[0])/vox_size)
        sy = np.arange(bd_min[1], bd_max[1], (bd_max[1] - bd_min[1])/vox_size)
        sz = np.arange(bd_min[2], bd_max[2], (bd_max[2] - bd_min[2])/vox_size)
        print(bd_max)
        print(bd_min)
        for i in sx:
            for j in sy:
                for k in sz:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        input_points_2d_bs = input_points_2d_bs.reshape((vox_size,vox_size,vox_size,3))
        
                    
        vox = []
        feature_bs = []
        for j in range(vox_size*vox_size):
            t = np.zeros(SHAPE_NUM)
            t[0] = 1
            feature_bs.append(t)
        feature_bs = np.asarray(feature_bs)
        for i in range(vox_size):
            
            input_points_2d_bs_t = input_points_2d_bs[i,:,:,:]
            input_points_2d_bs_t = input_points_2d_bs_t.reshape(vox_size*vox_size, 3)
            #print(input_points_2d_bs_t.shape)
            feature_bs_t = feature_bs.reshape(vox_size*vox_size,SHAPE_NUM)
            #print(feature_bs_t.shape)
            sdf_c = sess.run([sdf],feed_dict={input_points_3d:input_points_2d_bs_t,feature:feature_bs_t})
            vox.append(sdf_c)
            
        vox = np.asarray(vox)
        #print('vox',vox.shape)
        
        
        
        vox = vox.reshape((vox_size,vox_size,vox_size))
        
        
        #threshs = [0.005]
        tn = 0
        #threshs = [0.0009,0.001,0.0011,0.0012,0.0013,0.0014,0.0015,0.0016,0.0017]
        threshs = [0.002,0.0025,0.005,0.01]
        # threshs = [0.002,0.005,0.01]
        
        for thresh in threshs:
            print(np.sum(vox>thresh),np.sum(vox<thresh))
            
            if(np.sum(vox>0.0) < np.sum(vox<0.0)):
                thresh = -thresh
            vertices, triangles = libmcubes.marching_cubes(vox, thresh)
            if(vertices.shape[0]<10 or triangles.shape[0]<10):
                print('no sur---------------------------------------------')
                continue
            if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                triangles_t = []
                for it in range(triangles.shape[0]):
                    tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                    triangles_t.append(tt)
                triangles_t = np.asarray(triangles_t)
            else:
                triangles_t = triangles
                triangles_t = np.asarray(triangles_t)

            vertices -= 0.5
            # Undo padding
            vertices -= 1
            # Normalize to bounding box
            vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
            vertices = (bd_max-bd_min) * vertices + bd_min

            vertices = vertices * loc_data['scal'] + loc_data['trans']
            
            mesh = trimesh.Trimesh(vertices, triangles_t,
                            vertex_normals=None,
                            process=False)
            mesh.export(OUTPUT_DIR +  '/occn_' + a.dataname + '_'+ str(thresh) + '.obj')
        
        split_num = point_sparse.shape[0]//(vox_size*vox_size)
        denoise = []
        for i in range(split_num):
            
            input_points_2d_bs_t = point_sparse[i*vox_size*vox_size:(i+1)*vox_size*vox_size]
            input_points_2d_bs_t = input_points_2d_bs_t.reshape(vox_size*vox_size, 3)
            #print(input_points_2d_bs_t.shape)
            feature_bs_t = feature_bs.reshape(vox_size*vox_size,SHAPE_NUM)
            #print(feature_bs_t.shape)
            points_gen_c = sess.run([points_gen],feed_dict={input_points_3d:input_points_2d_bs_t,feature:feature_bs_t})
            denoise.append(points_gen_c)
        denoise = np.asarray(denoise).reshape(-1,3) * loc_data['scal'] + loc_data['trans']
        vis_single_points(denoise, OUTPUT_DIR +  '/denoise_' + a.dataname + '.ply')
        
        
        
        
        
    
                    
    
    