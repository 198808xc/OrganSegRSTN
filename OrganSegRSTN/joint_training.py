import numpy as np
import os
import sys
import shutil
import time
import urllib
import caffe
from utils import *
import surgery


data_path = sys.argv[1]
current_fold = int(sys.argv[2])
organ_number = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])
slice_threshold = float(sys.argv[6])
slice_thickness = int(sys.argv[7])
organ_ID = int(sys.argv[8])
plane = sys.argv[9]
GPU_ID = int(sys.argv[10])
learning_rate1 = float(sys.argv[11])
learning_rate_m1 = int(sys.argv[12])
learning_rate2 = float(sys.argv[13])
learning_rate_m2 = int(sys.argv[14])
margin = int(sys.argv[15])
prob = float(sys.argv[16])
sample_batch = int(sys.argv[17])
snapshot_path1 = os.path.join(snapshot_path, 'indiv:' + \
    sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + str(margin))
snapshot_path2 = os.path.join(snapshot_path, 'joint:' + \
    sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + \
    sys.argv[13] + 'x' + str(learning_rate_m2) + ',' + str(margin))
step = int(sys.argv[18])
max_iterations1 = int(sys.argv[19])
max_iterations2 = int(sys.argv[20])
timestamp1 = sys.argv[21]
if len(sys.argv) == 23:
    timestamp2 = sys.argv[22]

if __name__ == '__main__':
    if len(timestamp1) < 15:
        snapshot_name = snapshot_name_from_timestamp(snapshot_path1, \
            current_fold, plane, 'I', slice_thickness, organ_ID, [max_iterations1], timestamp1)
        timestamp1 = snapshot_name[-15: ]
    snapshot_name2 = 'FD' + str(current_fold) + ':' + \
        plane + 'I' + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp1 + ',' + \
        plane + 'J' + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp2
    snapshot_directory2 = os.path.join(snapshot_path2, snapshot_name2)
    if not os.path.exists(snapshot_directory2):
        os.makedirs(snapshot_directory2)
    snapshot_name2_ = 'FD' + str(current_fold) + ':' + \
        plane + 'J' + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp2
    log_file2 = os.path.join(log_path, snapshot_name2_ + '.txt')
    log_file2_ = log_filename(snapshot_directory2)


if __name__ == '__main__':
    if not os.path.exists(prototxt_path):
        os.makedirs(prototxt_path)
    prototxt_filename = 'training_J' + str(slice_thickness) + \
        'x' + str(learning_rate_m2) + '.prototxt'
    prototxt_file = os.path.join(prototxt_path, prototxt_filename)
    prototxt_file_ = os.path.join('prototxts', prototxt_filename)
    shutil.copyfile(prototxt_file_, prototxt_file)
    solver_filename = 'solver_J' + str(slice_thickness) + \
        '_FD' + str(current_fold) + '.prototxt'
    solver_file = os.path.join(prototxt_path, solver_filename)
    output = open(solver_file, 'w')
    output.write('train_net: \"' + prototxt_file + '\"\n')
    output.write('\n' * 1)
    output.write('display: 20\n')
    output.write('average_loss: 20\n')
    output.write('\n' * 1)
    output.write('base_lr: ' + str(learning_rate2) + '\n')
    output.write('lr_policy: \"multistep\"\n')
    output.write('gamma: 0.5\n')
    output.write('stepvalue: ' + str(max_iterations1 + max_iterations2 / 4 * 1) + '\n')
    output.write('stepvalue: ' + str(max_iterations1 + max_iterations2 / 4 * 2) + '\n')
    output.write('stepvalue: ' + str(max_iterations1 + max_iterations2 / 4 * 3) + '\n')
    output.write('\n' * 1)
    output.write('momentum: 0.99\n')
    output.write('\n' * 1)
    output.write('iter_size: 1\n')
    output.write('weight_decay: 0.0005\n')
    output.write('snapshot: ' + str(step) + '\n')
    output.write('snapshot_prefix: \"' + os.path.join(snapshot_directory2, 'train') + '\"\n')
    output.write('\n' * 1)
    output.write('test_initialization: false\n')
    output.close()
    sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    snapshot_name = snapshot_name_from_timestamp(snapshot_path1, \
        current_fold, plane, 'I', slice_thickness, organ_ID, [max_iterations1], timestamp1)
    snapshot_ = os.path.join(snapshot_path1, \
        snapshot_name, 'train_iter_' + str(max_iterations1) + '.solverstate')
    solver.restore(snapshot_)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)
    solver.step(max_iterations2)
    shutil.copyfile(log_file2, log_file2_)
