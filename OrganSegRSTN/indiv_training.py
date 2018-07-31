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
snapshot_path = os.path.join(snapshot_path, 'indiv:' + \
    sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + str(margin))
step = int(sys.argv[18])
max_iterations1 = int(sys.argv[19])
max_iterations2 = int(sys.argv[20])
fraction = float(sys.argv[21])
separate_iterations = max(int(round(max_iterations1 * fraction / step)), 1) * step
timestamp = sys.argv[22]

if __name__ == '__main__':
    snapshot_name = 'FD' + str(current_fold) + ':' + \
        plane + 'I' + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
    snapshot_directory = os.path.join(snapshot_path, snapshot_name)
    if not os.path.exists(snapshot_directory):
        os.makedirs(snapshot_directory)
    log_file = os.path.join(log_path, snapshot_name + '.txt')
    log_file_ = log_filename(snapshot_directory)
    weights = os.path.join(pretrained_model_path, 'RSTN-scratch.caffemodel')
    if not os.path.isfile(weights):
        sys.exit('Error: the scratch model was not found, please download it from our GitHub.')


if __name__ == '__main__':
    if not os.path.exists(prototxt_path):
        os.makedirs(prototxt_path)
    while True:
        if fraction > 0:
            prototxt_filename = 'training_S' + str(slice_thickness) + \
                'x' + str(learning_rate_m1) + '.prototxt'
        else:
            prototxt_filename = 'training_I' + str(slice_thickness) + \
                'x' + str(learning_rate_m1) + '.prototxt'
        prototxt_file = os.path.join(prototxt_path, prototxt_filename)
        prototxt_file_ = os.path.join('prototxts', prototxt_filename)
        shutil.copyfile(prototxt_file_, prototxt_file)
        if fraction > 0:
            solver_filename = 'solver_S' + str(slice_thickness) + \
                '_FD' + str(current_fold) + '.prototxt'
        else:
            solver_filename = 'solver_I' + str(slice_thickness) + \
                '_FD' + str(current_fold) + '.prototxt'
        solver_file = os.path.join(prototxt_path, solver_filename)
        output = open(solver_file, 'w')
        output.write('train_net: \"' + prototxt_file + '\"\n')
        output.write('\n' * 1)
        output.write('display: 20\n')
        output.write('average_loss: 20\n')
        output.write('\n' * 1)
        output.write('base_lr: ' + str(learning_rate1) + '\n')
        output.write('lr_policy: \"fixed\"\n')
        output.write('stepvalue: ' + str(max_iterations1) + '\n')
        output.write('\n' * 1)
        output.write('momentum: 0.99\n')
        output.write('\n' * 1)
        output.write('iter_size: 1\n')
        output.write('weight_decay: 0.0005\n')
        output.write('snapshot: ' + str(step) + '\n')
        output.write('snapshot_prefix: \"' + os.path.join(snapshot_directory, 'train') + '\"\n')
        output.write('\n' * 1)
        output.write('test_initialization: false\n')
        output.close()
        sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
        caffe.set_device(GPU_ID)
        caffe.set_mode_gpu()
        solver = caffe.SGDSolver(solver_file)
        solver.net.copy_from(weights)
        interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
        surgery.interp(solver.net, interp_layers)
        solver.step(separate_iterations)
        if valid_loss(log_file, separate_iterations):
            break
    prototxt_filename = 'training_I' + str(slice_thickness) + \
        'x' + str(learning_rate_m1) + '.prototxt'
    prototxt_file = os.path.join(prototxt_path, prototxt_filename)
    prototxt_file_ = os.path.join('prototxts', prototxt_filename)
    shutil.copyfile(prototxt_file_, prototxt_file)
    solver_filename = 'solver_I' + str(slice_thickness) + \
        '_FD' + str(current_fold) + '.prototxt'
    solver_file = os.path.join(prototxt_path, solver_filename)
    output = open(solver_file, 'w')
    output.write('train_net: \"' + prototxt_file + '\"\n')
    output.write('\n' * 1)
    output.write('display: 20\n')
    output.write('average_loss: 20\n')
    output.write('\n' * 1)
    output.write('base_lr: ' + str(learning_rate1) + '\n')
    output.write('lr_policy: \"fixed\"\n')
    output.write('stepvalue: ' + str(max_iterations1) + '\n')
    output.write('\n' * 1)
    output.write('momentum: 0.99\n')
    output.write('\n' * 1)
    output.write('iter_size: 1\n')
    output.write('weight_decay: 0.0005\n')
    output.write('snapshot: ' + str(step) + '\n')
    output.write('snapshot_prefix: \"' + os.path.join(snapshot_directory, 'train') + '\"\n')
    output.write('\n' * 1)
    output.write('test_initialization: false\n')
    output.close()
    sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    snapshot_ = os.path.join(snapshot_directory, \
        'train_iter_' + str(separate_iterations) + '.solverstate')
    solver.restore(snapshot_)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)
    solver.step(max_iterations1 - separate_iterations)
    shutil.copyfile(log_file, log_file_)
