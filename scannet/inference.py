import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
import tf_util
import pc_util
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import scannet_dataset
import open3d as o3d
import cv2
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 21

# Shapenet official test split
# DATA_PATH = os.path.join(ROOT_DIR,'data','scannet_data_pointnet2')
# TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')

def loadCustomDataset(path):
    o3dPly = o3d.io.read_point_cloud(path)
    npPts = np.asarray(o3dPly.points)

    if (npPts.shape[0] < NUM_POINT):
        log_string('Input mesh needs to be larger than %s points' % NUM_POINT)
        quit()

    o3dPts = o3d.geometry.PointCloud()
    o3dPts.points = o3d.utility.Vector3dVector(npPts)
    o3d.visualization.draw_geometries([o3dPts])

    numBatches = npPts.shape[0] / NUM_POINT
    numBatchPts = numBatches*NUM_POINT
    numLeftover = npPts.shape[0] % NUM_POINT

    batchedPts = npPts[:numBatchPts,:]
    leftoverPts = npPts[numBatchPts:numBatchPts+numLeftover,:]
    fillerPts = npPts[numBatchPts-NUM_POINT+numLeftover:numBatchPts,:]

    batchedPts = np.concatenate((batchedPts, fillerPts, leftoverPts), axis=0)
    batchedPts = np.reshape(batchedPts, (numBatches+1, NUM_POINT, 3))

    batchedLabels = np.ones((batchedPts.shape[0], batchedPts.shape[1]))
    batchedSmpw = np.ones((batchedPts.shape[0], batchedPts.shape[1]))

    return (batchedPts, batchedLabels, batchedSmpw)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print is_training_pl

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print "--- Get model and loss"
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        # sess = tf.Session(config=config)
        sess = tf.Session(config=config)
        saver.restore(sess, 'log/run1/best_model_epoch_120.ckpt')
        log_string('--- Loaded weights from log/run1/best_model_epoch_120.ckpt')

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        # init = tf.global_variables_initializer()
        # sess.run(init, {is_training_pl: True})
        # sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
              'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            eval_whole_scene_one_epoch(sess, ops, test_writer)

# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----'%(EPOCH_CNT))

    labelweights = np.zeros(21)
    labelweights_vox = np.zeros(21)

    batch_data, batch_label, batch_smpw = loadCustomDataset(path="../data/meshFinal1.ply")

    while batch_data.shape[0] < BATCH_SIZE:
        batch_data = np.concatenate((batch_data, batch_data))
        batch_label = np.concatenate((batch_label, batch_label))
        batch_smpw = np.concatenate((batch_smpw, batch_smpw))
    
    for iter in range(batch_data.shape[0]/BATCH_SIZE):

        vizMat = np.zeros([0,3])
        vizLabels = np.zeros([0,1])

        startBatch = iter*BATCH_SIZE
        in_data = batch_data[startBatch:startBatch+BATCH_SIZE,:,:]
        in_label = batch_label[startBatch:startBatch+BATCH_SIZE,:]
        in_smpw = batch_smpw[startBatch:startBatch+BATCH_SIZE,:]

        feed_dict = {ops['pointclouds_pl']: in_data,
                     ops['labels_pl']: in_label,
                     ops['smpws_pl']: in_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == in_label) & (in_label>0) & (in_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((in_label>0) & (in_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(in_label,range(22))
        labelweights += tmp
   
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((in_label==l) & (in_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (in_label==l) & (in_smpw>0))

        for b in xrange(in_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(in_data[b,in_smpw[b,:]>0,:], np.concatenate((np.expand_dims(in_label[b,in_smpw[b,:]>0],1),np.expand_dims(pred_val[b,in_smpw[b,:]>0],1)),axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
            tmp,_ = np.histogram(uvlabel[:,0],range(22))
            labelweights_vox += tmp

            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

            vizMat = np.concatenate((vizMat, in_data[b,in_smpw[b,:]>0,:]))
            vizLabels = np.concatenate((vizLabels, np.expand_dims(pred_val[b,in_smpw[b,:]>0],1)))

        numLabels = 21
        vizLabels = vizLabels.flatten().tolist()
        labelSet = [int(i) for i in list(set(vizLabels))]
        
        # print(labelSet)

        # rVals = random.sample(np.linspace(0, 1.0, num=numLabels), numLabels)
        # gVals = random.sample(np.linspace(0, 1.0, num=numLabels), numLabels)
        # bVals = random.sample(np.linspace(0, 1.0, num=numLabels), numLabels)

        # colorDict = {}
        # for i in xrange(numLabels):
        #     colorDict[i] = [rVals[i], gVals[i], bVals[i]]

        # # NYU-40 Original
        # colorDict = {
        #     0: [0.19047619047619047, 0.09523809523809523, 0.0],
        #     1: [0.0, 0.14285714285714285, 0.8095238095238095],
        #     2: [1.0, 0.8095238095238095, 0.38095238095238093],
        #     3: [0.47619047619047616, 0.38095238095238093, 1.0],
        #     4: [0.8571428571428571, 0.5714285714285714, 0.19047619047619047],
        #     5: [0.5238095238095237, 0.6190476190476191, 0.3333333333333333],
        #     6: [0.8095238095238095, 0.9047619047619047, 0.8571428571428571],
        #     7: [0.6190476190476191, 1.0, 0.09523809523809523],
        #     8: [0.42857142857142855, 0.5238095238095237, 0.047619047619047616],
        #     9: [0.5714285714285714, 0.23809523809523808, 0.14285714285714285],
        #     10: [0.2857142857142857, 0.2857142857142857, 0.6190476190476191],
        #     11: [0.047619047619047616, 0.9523809523809523, 0.7142857142857142],
        #     12: [0.3333333333333333, 0.7619047619047619, 0.7619047619047619],
        #     13: [0.09523809523809523, 0.42857142857142855, 0.47619047619047616],
        #     14: [0.9523809523809523, 0.6666666666666666, 0.42857142857142855],
        #     15: [0.38095238095238093, 0.0, 0.2857142857142857],
        #     16: [0.6666666666666666, 0.047619047619047616, 0.23809523809523808],
        #     17: [0.7619047619047619, 0.19047619047619047, 0.5238095238095237],
        #     18: [0.7142857142857142, 0.7142857142857142, 0.9047619047619047],
        #     19: [0.23809523809523808, 0.47619047619047616, 0.5714285714285714],
        #     20: [0.14285714285714285, 0.8571428571428571, 0.6666666666666666],
        #     21: [0.9047619047619047, 0.3333333333333333, 0.9523809523809523]
        # }

        # # NYU-40 Original
        # labelDict = {
        #     0: 'undefined',
        #     1: 'wall',
        #     2: 'floor',
        #     3: 'cabinet',
        #     4: 'bed',
        #     5: 'chair',
        #     6: 'sofa',
        #     7: 'table',
        #     8: 'door',
        #     9: 'window',
        #     10: 'bookshelf',
        #     11: 'picture',
        #     12: 'counter',
        #     13: 'desk',
        #     14: 'curtain',
        #     15: 'refridgerator',
        #     16: 'shower curtain',
        #     17: 'toilet',
        #     18: 'sink',
        #     19: 'batchtub',
        #     20: 'otherfurniture',
        # }

        # Custom Labels
        colorDict = {
            0: [0.19047619047619047, 0.09523809523809523, 0.0],
            5: [0.19047619047619047, 0.09523809523809523, 0.0],
            7: [0.19047619047619047, 0.09523809523809523, 0.0],
            9: [0.19047619047619047, 0.09523809523809523, 0.0],
            10: [0.19047619047619047, 0.09523809523809523, 0.0],
            11: [0.19047619047619047, 0.09523809523809523, 0.0],
            12: [0.19047619047619047, 0.09523809523809523, 0.0],
            13: [0.19047619047619047, 0.09523809523809523, 0.0],
            15: [0.19047619047619047, 0.09523809523809523, 0.0],
            16: [0.19047619047619047, 0.09523809523809523, 0.0],
            17: [0.19047619047619047, 0.09523809523809523, 0.0],
            18: [0.19047619047619047, 0.09523809523809523, 0.0],
            19: [0.19047619047619047, 0.09523809523809523, 0.0],
            1: [0.0, 0.14285714285714285, 0.8095238095238095],
            2: [1.0, 0.8095238095238095, 0.38095238095238093],
            3: [0.47619047619047616, 0.38095238095238093, 1.0],
            4: [0.8571428571428571, 0.5714285714285714, 0.19047619047619047],
            6: [0.8095238095238095, 0.9047619047619047, 0.8571428571428571],
            8: [0.42857142857142855, 0.5238095238095237, 0.047619047619047616],
            14: [0.9523809523809523, 0.6666666666666666, 0.42857142857142855],
            20: [0.14285714285714285, 0.8571428571428571, 0.6666666666666666]
        }

        # Custom Labels
        labelDict = {
            0: 'undefined',
            5: 'undefined',
            7: 'undefined',
            9: 'undefined',
            10: 'undefined',
            11: 'undefined',
            12: 'undefined',
            13: 'undefined',
            15: 'undefined',
            16: 'undefined',
            17: 'undefined',
            18: 'undefined',
            19: 'undefined',
            1: 'wall',
            2: 'floor',
            3: 'chair',
            4: 'table',
            6: 'bed',
            8: 'sofa',
            14: 'curtain/door/window',
            20: 'otherfurniture'
        }

        # leftStart = 50
        # hInc = 50
        # blackMat = np.zeros((2000,1000,3))
        # for i in range(numLabels):
        #     cv2.putText(blackMat, "%s" % labelDict[i], (leftStart, leftStart+(i*hInc)), cv2.FONT_HERSHEY_SIMPLEX, 2, (int(255.0*colorDict[i][2]), int(255.0*colorDict[i][1]), int(255.0*colorDict[i][0])), 5)
        # cv2.imwrite('../data/labelRef.png', blackMat)

        pointColors = [colorDict[label] for label in vizLabels]
        pointColors = np.array(pointColors)

        o3dViz = o3d.geometry.PointCloud()
        o3dViz.points = o3d.utility.Vector3dVector(vizMat)
        o3dViz.colors = o3d.utility.Vector3dVector(pointColors)
        o3d.visualization.draw_geometries([o3dViz])

    EPOCH_CNT += 1

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
