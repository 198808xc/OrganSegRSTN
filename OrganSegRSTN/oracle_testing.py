import numpy as np
import os
import sys
import shutil
import time
import caffe
from utils import *


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
result_path = os.path.join(result_path, 'oracle:' + \
    sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + \
    sys.argv[13] + 'x' + str(learning_rate_m2) + ',' + str(margin))
max_iterations1 = int(sys.argv[18])
max_iterations2 = int(sys.argv[19])
starting_iterations = int(sys.argv[20])
step = int(sys.argv[21])
max_iterations = int(sys.argv[22])
iteration = range(starting_iterations, max_iterations + 1, step)
timestamp1 = sys.argv[23]
timestamp2 = sys.argv[24]
[snapshot_name1, snapshot_name2] = snapshot_name_from_timestamp_2(snapshot_path1, snapshot_path2, \
    current_fold, plane, 'I', 'J', slice_thickness, organ_ID, iteration, timestamp1, timestamp2)
if snapshot_name1 == '' or snapshot_name2 == '':
    exit('Error: no valid snapshot directories are detected!')
snapshot_directory1 = os.path.join(snapshot_path1, snapshot_name1)
print 'Snapshot directory 1: ' + snapshot_directory1 + ' .'
snapshot_directory2 = os.path.join(snapshot_path2, snapshot_name2)
print 'Snapshot directory 2: ' + snapshot_directory2 + ' .'
snapshot = []
for t in range(len(iteration)):
    snapshot_file1 = snapshot_filename(snapshot_directory1, iteration[t])
    snapshot_file2 = snapshot_filename(snapshot_directory2, iteration[t])
    if os.path.isfile(snapshot_file1):
        snapshot.append(snapshot_file1)
    else:
        snapshot.append(snapshot_file2)
print str(len(snapshot)) + ' snapshots are to be evaluated.'
for t in range(len(snapshot)):
    print '  Snapshot #' + str(t + 1) + ': ' + snapshot[t] + ' .'
result_name = snapshot_name2

sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()

volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
while volume_list[len(volume_list) - 1] == '':
    volume_list.pop()
DSC = np.zeros((len(snapshot), len(volume_list)))
result_directory = os.path.join(result_path, result_name, 'volumes')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
result_file = os.path.join(result_path, result_name, 'results.txt')
output = open(result_file, 'w')
output.close()
for t in range(len(snapshot)):
    output = open(result_file, 'a+')
    output.write('Evaluating snapshot ' + str(iteration[t]) + ':\n')
    output.close()
    finished = True
    for i in range(len(volume_list)):
        volume_file = volume_filename_testing(result_directory, iteration[t], i)
        if not os.path.isfile(volume_file):
            finished = False
            break
    if not finished:
        deploy_filename = 'deploy_' + 'O' + str(slice_thickness) + '.prototxt'
        deploy_file = os.path.join(prototxt_path, deploy_filename)
        deploy_file_ = os.path.join('prototxts', deploy_filename)
        shutil.copyfile(deploy_file_, deploy_file)
        net = caffe.Net(deploy_file, snapshot[t], caffe.TEST)
    for i in range(len(volume_list)):
        start_time = time.time()
        print 'Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases, ' + \
            str(t + 1) + ' out of ' + str(len(snapshot)) + ' snapshots.'
        volume_file = volume_filename_testing(result_directory, iteration[t], i)
        s = volume_list[i].split(' ')
        label = np.load(s[2])
        label = is_organ(label, organ_ID).astype(np.uint8)
        if not os.path.isfile(volume_file):
            image = np.load(s[1]).astype(np.float32)
            np.minimum(np.maximum(image, low_range, image), high_range, image)
            image -= low_range
            image /= (high_range - low_range)
            print '  Data loading is finished: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
            pred = np.zeros(image.shape, dtype = np.float32)
            label_sumX = np.sum(label, axis = (1, 2))
            label_sumY = np.sum(label, axis = (0, 2))
            label_sumZ = np.sum(label, axis = (0, 1))
            if label_sumX.sum() == 0:
                continue
            minR = 0
            if plane == 'X':
                maxR = image.shape[0]
                shape_ = (1, 3, image.shape[1], image.shape[2])
            elif plane == 'Y':
                maxR = image.shape[1]
                shape_ = (1, 3, image.shape[0], image.shape[2])
            elif plane == 'Z':
                maxR = image.shape[2]
                shape_ = (1, 3, image.shape[0], image.shape[1])
            for j in range(minR, maxR):
                if slice_thickness == 1:
                    sID = [j, j, j]
                elif slice_thickness == 3:
                    sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]
                if plane == 'X':
                    if label_sumX[sID].sum() == 0:
                        continue
                    net.blobs['data'].reshape(*shape_)
                    net.blobs['label'].reshape(*shape_)
                    net.blobs['data'].data[0, ...] = image[sID, :, :]
                    net.blobs['label'].data[0, ...] = label[sID, :, :]
                elif plane == 'Y':
                    if label_sumY[sID].sum() == 0:
                        continue
                    net.blobs['data'].reshape(*shape_)
                    net.blobs['label'].reshape(*shape_)
                    net.blobs['data'].data[0, ...] = image[:, sID, :].transpose(1, 0, 2)
                    net.blobs['label'].data[0, ...] = label[:, sID, :].transpose(1, 0, 2)
                elif plane == 'Z':
                    if label_sumZ[sID].sum() == 0:
                        continue
                    net.blobs['data'].reshape(*shape_)
                    net.blobs['label'].reshape(*shape_)
                    net.blobs['data'].data[0, ...] = image[:, :, sID].transpose(2, 0, 1)
                    net.blobs['label'].data[0, ...] = label[:, :, sID].transpose(2, 0, 1)
                net.blobs['crop_margin'].reshape((1))
                net.blobs['crop_margin'].data[...] = margin
                net.blobs['crop_prob'].reshape((1))
                net.blobs['crop_prob'].data[...] = 0
                net.blobs['crop_sample_batch'].reshape((1))
                net.blobs['crop_sample_batch'].data[...] = 0
                net.forward()
                out = net.blobs['prob-R'].data[0, :, :, :]
                if slice_thickness == 1:
                    if plane == 'X':
                        pred[j, :, :] = out
                    elif plane == 'Y':
                        pred[:, j, :] = out
                    elif plane == 'Z':
                        pred[:, :, j] = out
                elif slice_thickness == 3:
                    if plane == 'X':
                        if j == minR:
                            pred[j: j + 2, :, :] += out[1: 3, :, :]
                        elif j == maxR - 1:
                            pred[j - 1: j + 1, :, :] += out[0: 2, :, :]
                        else:
                            pred[j - 1: j + 2, :, :] += out[...]
                    elif plane == 'Y':
                        if j == minR:
                            pred[:, j: j + 2, :] += out[1: 3, :, :].transpose(1, 0, 2)
                        elif j == maxR - 1:
                            pred[:, j - 1: j + 1, :] += out[0: 2, :, :].transpose(1, 0, 2)
                        else:
                            pred[:, j - 1: j + 2, :] += out[...].transpose(1, 0, 2)
                    elif plane == 'Z':
                        if j == minR:
                            pred[:, :, j: j + 2] += out[1: 3, :, :].transpose(1, 2, 0)
                        elif j == maxR - 1:
                            pred[:, :, j - 1: j + 1] += out[0: 2, :, :].transpose(1, 2, 0)
                        else:
                            pred[:, :, j - 1: j + 2] += out[...].transpose(1, 2, 0)
            if slice_thickness == 3:
                if plane == 'X':
                    pred[minR, :, :] /= 2
                    pred[minR + 1: maxR - 1, :, :] /= 3
                    pred[maxR - 1, :, :] /= 2
                elif plane == 'Y':
                    pred[:, minR, :] /= 2
                    pred[:, minR + 1: maxR - 1, :] /= 3
                    pred[:, maxR - 1, :] /= 2
                elif plane == 'Z':
                    pred[:, :, minR] /= 2
                    pred[:, :, minR + 1: maxR - 1] /= 3
                    pred[:, :, maxR - 1] /= 2
            print '  Testing is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.'
            pred = np.around(pred * 255).astype(np.uint8)
            np.savez_compressed(volume_file, volume = pred)
            print '  Data saving is finished: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
            pred_temp = (pred >= 128)
        else:
            pred = np.load(volume_file)['volume'].astype(np.uint8)
            print '  Testing result is loaded: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
            pred_temp = (pred >= 128)
        DSC[t, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_temp)
        print '    DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
            ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .'
        output = open(result_file, 'a+')
        output.write('  Testcase ' + str(i + 1) + ': DSC = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC[t, i] = 0
        print '  DSC computation is finished: ' + \
            str(time.time() - start_time) + ' second(s) elapsed.'
    print 'Snapshot ' + str(iteration[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .'
    output = open(result_file, 'a+')
    output.write('Snapshot ' + str(iteration[t]) + \
        ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .\n')
    output.close()
print 'The testing process is finished.'
for t in range(len(snapshot)):
    print '  Snapshot ' + str(iteration[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .'
