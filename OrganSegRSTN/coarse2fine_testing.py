import numpy as np
import os
import sys
import shutil
import time
import caffe
from utils import *
import scipy.io


data_path = sys.argv[1]
current_fold = int(sys.argv[2])
organ_number = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])
slice_threshold = float(sys.argv[6])
slice_thickness = int(sys.argv[7])
organ_ID = int(sys.argv[8])
GPU_ID = int(sys.argv[9])
learning_rate1 = float(sys.argv[10])
learning_rate_m1 = int(sys.argv[11])
learning_rate2 = float(sys.argv[12])
learning_rate_m2 = int(sys.argv[13])
margin = int(sys.argv[14])
fine_snapshot_path1 = os.path.join(snapshot_path, 'indiv:' + \
    sys.argv[10] + 'x' + str(learning_rate_m1))
fine_snapshot_path2 = os.path.join(snapshot_path, 'joint:' + \
    sys.argv[10] + 'x' + str(learning_rate_m1) + ',' + \
    sys.argv[12] + 'x' + str(learning_rate_m2) + ',' + str(margin))
coarse_result_path = os.path.join(result_path, 'coarse:' + \
    sys.argv[10] + 'x' + str(learning_rate_m1) + ',' + \
    sys.argv[12] + 'x' + str(learning_rate_m2) + ',' + str(margin))
coarse2fine_result_path = os.path.join(result_path, 'coarse2fine:' + \
    sys.argv[10] + 'x' + str(learning_rate_m1) + ',' + \
    sys.argv[12] + 'x' + str(learning_rate_m2) + ',' + str(margin))
coarse_starting_iterations = int(sys.argv[15])
coarse_step = int(sys.argv[16])
coarse_max_iterations = int(sys.argv[17])
coarse_iteration = range(coarse_starting_iterations, coarse_max_iterations + 1, coarse_step)
coarse_threshold = float(sys.argv[18])
fine_starting_iterations = int(sys.argv[19])
fine_step = int(sys.argv[20])
fine_max_iterations = int(sys.argv[21])
fine_iteration = range(fine_starting_iterations, fine_max_iterations + 1, fine_step)
fine_threshold = float(sys.argv[22])
max_rounds = int(sys.argv[23])
timestamp1 = {}
timestamp1['X'] = sys.argv[24]
timestamp1['Y'] = sys.argv[25]
timestamp1['Z'] = sys.argv[26]
timestamp2 = {}
timestamp2['X'] = sys.argv[27]
timestamp2['Y'] = sys.argv[28]
timestamp2['Z'] = sys.argv[29]
volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
while volume_list[len(volume_list) - 1] == '':
    volume_list.pop()

print 'Looking for snapshots:'
fine_snapshot_ = {}
fine_snapshot_name_ = {}
for plane in ['X', 'Y', 'Z']:
    [fine_snapshot_name1, fine_snapshot_name2] = snapshot_name_from_timestamp_2( \
        fine_snapshot_path1, fine_snapshot_path2, current_fold, plane, 'I', 'J', \
        slice_thickness, organ_ID, fine_iteration, timestamp1[plane], timestamp2[plane])
    if fine_snapshot_name1 == '' or fine_snapshot_name2 == '':
        exit('  Error: no valid snapshot directories are detected!')
    fine_snapshot_directory1 = os.path.join(fine_snapshot_path1, fine_snapshot_name1)
    print '  Snapshot directory 1 for plane ' + plane + ': ' + fine_snapshot_directory1 + ' .'
    fine_snapshot_directory2 = os.path.join(fine_snapshot_path2, fine_snapshot_name2)
    print '  Snapshot directory 2 for plane ' + plane + ': ' + fine_snapshot_directory2 + ' .'
    fine_snapshot = []
    for t in range(len(fine_iteration)):
        fine_snapshot_file1 = snapshot_filename(fine_snapshot_directory1, fine_iteration[t])
        fine_snapshot_file2 = snapshot_filename(fine_snapshot_directory2, fine_iteration[t])
        if os.path.isfile(fine_snapshot_file1):
            fine_snapshot.append(fine_snapshot_file1)
        else:
            fine_snapshot.append(fine_snapshot_file2)
    print '  ' + str(len(fine_snapshot)) + ' snapshots are to be evaluated.'
    for t in range(len(fine_snapshot)):
        print '    Snapshot #' + str(t + 1) + ': ' + fine_snapshot[t] + ' .'
    fine_snapshot_[plane] = fine_snapshot
    fine_snapshot_name2 = fine_snapshot_name2.split(':')[1]
    fine_snapshot_name_[plane] = fine_snapshot_name2.split(',')
    timestamp1[plane] = fine_snapshot_name_[plane][0][-15: ]
    timestamp2[plane] = fine_snapshot_name_[plane][1][-15: ]

print 'In the coarse stage:'
coarse_result_name_ = {}
coarse_result_directory_ = {}
for plane in ['X', 'Y', 'Z']:
    coarse_result_name__ = result_name_from_timestamp_2(coarse_result_path, \
        current_fold, plane, 'I', 'J', slice_thickness, organ_ID, \
        coarse_iteration, volume_list, timestamp1[plane], timestamp2[plane])
    if coarse_result_name__ == '':
        exit('  Error: no valid result directories are detected!')
    coarse_result_directory__ = os.path.join(coarse_result_path, coarse_result_name__, 'volumes')
    print '  Result directory for plane ' + plane + ': ' + coarse_result_directory__ + ' .'
    if coarse_result_name__.startswith('FD'):
        index_ = coarse_result_name__.find(':')
        coarse_result_name__ = coarse_result_name__[index_ + 1: ]
    coarse_result_name_[plane] = coarse_result_name__
    coarse_result_directory_[plane] = coarse_result_directory__

coarse2fine_result_name = 'FD' + str(current_fold) + ':' + \
    fine_snapshot_name_['X'][0] + ',' + fine_snapshot_name_['X'][1] + ',' + \
    fine_snapshot_name_['Y'][0] + ',' + fine_snapshot_name_['Y'][1] + ',' + \
    fine_snapshot_name_['Z'][0] + ',' + fine_snapshot_name_['Z'][1] + ':' + \
    str(coarse_starting_iterations) + '_' + str(coarse_step) + '_' + \
    str(coarse_max_iterations) + ',' + str(coarse_threshold) + ':' + \
    str(fine_starting_iterations) + '_' + str(fine_step) + '_' + \
    str(fine_max_iterations) + ',' + str(fine_threshold) + ',' + str(max_rounds)
coarse2fine_result_directory = os.path.join( \
    coarse2fine_result_path, coarse2fine_result_name, 'volumes')
finished = np.ones((len(volume_list)), dtype = np.int)
for i in range(len(volume_list)):
    for r in range(max_rounds + 1):
        volume_file = volume_filename_coarse2fine(coarse2fine_result_directory, r, i)
        if not os.path.isfile(volume_file):
            finished[i] = 0
            break
finished_all = (finished.sum() == len(volume_list))
if finished_all:
    exit()
else:
    deploy_filename = 'deploy_' + 'F' + str(slice_thickness) + '.prototxt'
    deploy_file = os.path.join(prototxt_path, deploy_filename)
    deploy_file_ = os.path.join('prototxts', deploy_filename)
    shutil.copyfile(deploy_file_, deploy_file)

sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
net_ = {}
for plane in ['X', 'Y', 'Z']:
    net_[plane] = []
    for t in range(len(fine_iteration)):
        net = caffe.Net(deploy_file, fine_snapshot_[plane][t], caffe.TEST)
        net_[plane].append(net)
DSC = np.zeros((max_rounds + 1, len(volume_list)))
DSC_90 = np.zeros((len(volume_list)))
DSC_95 = np.zeros((len(volume_list)))
DSC_98 = np.zeros((len(volume_list)))
DSC_99 = np.zeros((len(volume_list)))
coarse2fine_result_directory = os.path.join(coarse2fine_result_path, \
    coarse2fine_result_name, 'volumes')
if not os.path.exists(coarse2fine_result_directory):
    os.makedirs(coarse2fine_result_directory)
coarse2fine_result_file = os.path.join(coarse2fine_result_path, \
    coarse2fine_result_name, 'results.txt')
output = open(coarse2fine_result_file, 'w')
output.close()
output = open(coarse2fine_result_file, 'a+')
output.write('Fusing results of ' + str(len(coarse_iteration)) + \
    ' and ' + str(len(fine_iteration)) + ' snapshots:\n')
output.close()

for i in range(len(volume_list)):
    start_time = time.time()
    print 'Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases.'
    output = open(coarse2fine_result_file, 'a+')
    output.write('  Testcase ' + str(i + 1) + ':\n')
    output.close()
    s = volume_list[i].split(' ')
    label = np.load(s[2])
    label = is_organ(label, organ_ID).astype(np.uint8)
    finished = True
    for r in range(max_rounds + 1):
        volume_file = volume_filename_coarse2fine(coarse2fine_result_directory, r, i)
        if not os.path.isfile(volume_file):
            finished = False
            break
    if not finished:
        image = np.load(s[1])
    print '  Data loading is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.'
    for r in range(max_rounds + 1):
        print '  Iteration round ' + str(r) + ':'
        volume_file = volume_filename_coarse2fine(coarse2fine_result_directory, r, i)
        if not finished:
            pred_ = np.zeros_like(label, dtype = np.float32)
            if r == 0:
                for plane in ['X', 'Y', 'Z']:
                    for t in range(len(coarse_iteration)):
                        volume_file_ = volume_filename_testing( \
                            coarse_result_directory_[plane], coarse_iteration[t], i)
                        volume_data = np.load(volume_file_)
                        pred_temp = volume_data['volume']
                        pred_ = pred_ + pred_temp
                pred_ = pred_ / 255 / len(coarse_iteration) / 3
                print '    Fusion is finished: ' + \
                    str(time.time() - start_time) + ' second(s) elapsed.'
            else:
                if mask.sum() == 0:
                    continue
                pred_ = np.zeros_like(label, dtype = np.float32)
                for plane in ['X', 'Y', 'Z']:
                    for t in range(len(fine_iteration)):
                        pred__ = np.zeros_like(image, dtype = np.float32)
                        net = net_[plane][t]
                        minR = 0
                        if plane == 'X':
                            maxR = label.shape[0]
                        elif plane == 'Y':
                            maxR = label.shape[1]
                        elif plane == 'Z':
                            maxR = label.shape[2]
                        for j in range(minR, maxR):
                            if slice_thickness == 1:
                                sID = [j, j, j]
                            elif slice_thickness == 3:
                                sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]
                            if plane == 'X':
                                image_ = image[sID, :, :].astype(np.float32)
                                score_ = score[sID, :, :].astype(np.float32)
                                mask_ = mask[sID, :, :].astype(np.uint8)
                            elif plane == 'Y':
                                image_ = image[:, sID, :].transpose(1, 0, 2).astype(np.float32)
                                score_ = score[:, sID, :].transpose(1, 0, 2).astype(np.float32)
                                mask_ = mask[:, sID, :].transpose(1, 0, 2).astype(np.uint8)
                            elif plane == 'Z':
                                image_ = image[:, :, sID].transpose(2, 0, 1).astype(np.float32)
                                score_ = score[:, :, sID].transpose(2, 0, 1).astype(np.float32)
                                mask_ = mask[:, :, sID].transpose(2, 0, 1).astype(np.uint8)
                            if mask_.sum() == 0:
                                continue
                            image_[image_ > high_range] = high_range
                            image_[image_ < low_range] = low_range
                            image_ = (image_ - low_range) / (high_range - low_range)
                            image_ = image_.reshape(1, \
                                image_.shape[0], image_.shape[1], image_.shape[2])
                            score_ = score_.reshape(1, \
                                score_.shape[0], score_.shape[1], score_.shape[2])
                            mask_ = mask_.reshape(1, \
                                mask_.shape[0], mask_.shape[1], mask_.shape[2])
                            net.blobs['data'].reshape(*image_.shape)
                            net.blobs['data'].data[...] = image_
                            net.blobs['prob'].reshape(*score_.shape)
                            net.blobs['prob'].data[...] = score_
                            net.blobs['label'].reshape(*mask_.shape)
                            net.blobs['label'].data[...] = mask_
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
                                    pred__[j, :, :] = out
                                elif plane == 'Y':
                                    pred__[:, j, :] = out
                                elif plane == 'Z':
                                    pred__[:, :, j] = out
                            elif slice_thickness == 3:
                                if plane == 'X':
                                    pred__[max(minR, j - 1): min(maxR, j + 2), :, :] += \
                                        out[max(minR, 1 - j): min(3, maxR + 1 - j), :, :]
                                elif plane == 'Y':
                                    pred__[:, max(minR, j - 1): min(maxR, j + 2), :] += \
                                        out[max(minR, 1 - j): min(3, maxR + 1 - j), \
                                            :, :].transpose(1, 0, 2)
                                elif plane == 'Z':
                                    pred__[:, :, max(minR, j - 1): min(maxR, j + 2)] += \
                                        out[max(minR, 1 - j): min(3, maxR + 1 - j), \
                                            :, :].transpose(1, 2, 0)
                        if slice_thickness == 3:
                            if plane == 'X':
                                pred__[minR, :, :] /= 2
                                pred__[minR + 1: maxR - 1, :, :] /= 3
                                pred__[maxR - 1, :, :] /= 2
                            elif plane == 'Y':
                                pred__[:, minR, :] /= 2
                                pred__[:, minR + 1: maxR - 1, :] /= 3
                                pred__[:, maxR - 1, :] /= 2
                            elif plane == 'Z':
                                pred__[:, :, minR] /= 2
                                pred__[:, :, minR + 1: maxR - 1] /= 3
                                pred__[:, :, maxR - 1] /= 2
                        print '    Testing on plane ' + plane + ' and snapshot ' + str(t + 1) + \
                            ' is finished: ' + str(time.time() - start_time) + \
                            ' second(s) elapsed.'
                        pred_ = pred_ + pred__
                pred_ = pred_ / len(fine_iteration) / 3
                print '    Testing is finished: ' + \
                    str(time.time() - start_time) + ' second(s) elapsed.'
            pred = np.zeros_like(pred_, dtype = np.int8)
            pred[pred_ >= fine_threshold] = 1
            if r > 0:
                pred = post_processing(pred, pred, 0.5, organ_ID)
            np.savez_compressed(volume_file, volume = pred)
        else:
            volume_data = np.load(volume_file)
            pred = volume_data['volume']
            print '    Testing result is loaded: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
        DSC[r, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
        print '      DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
            str(label_sum) + ') = ' + str(DSC[r, i]) + ' .'
        output = open(coarse2fine_result_file, 'a+')
        output.write('    Round ' + str(r) + ', ' + 'DSC = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[r, i]) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC[r, i] = 0
        if r > 0:
            inter_DSC, inter_sum, pred_sum, label_sum = DSC_computation(mask, pred)
            if pred_sum == 0 and label_sum == 0:
                inter_DSC = 1
            print '        Inter-iteration DSC = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(inter_DSC) + ' .'
            output = open(coarse2fine_result_file, 'a+')
            output.write('      Inter-iteration DSC = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(inter_DSC) + ' .\n')
            output.close()
            if DSC_90[i] == 0 and (r == max_rounds or inter_DSC >= 0.90):
                DSC_90[i] = DSC[r, i]
            if DSC_95[i] == 0 and (r == max_rounds or inter_DSC >= 0.95):
                DSC_95[i] = DSC[r, i]
            if DSC_98[i] == 0 and (r == max_rounds or inter_DSC >= 0.98):
                DSC_98[i] = DSC[r, i]
            if DSC_99[i] == 0 and (r == max_rounds or inter_DSC >= 0.99):
                DSC_99[i] = DSC[r, i]
        if r <= max_rounds:
            if not finished:
                score = np.copy(pred_)
            mask = np.copy(pred)
for r in range(max_rounds + 1):
    print 'Round ' + str(r) + ', ' + 'Average DSC = ' + str(np.mean(DSC[r, :])) + ' .'
    output = open(coarse2fine_result_file, 'a+')
    output.write('Round ' + str(r) + ', ' + 'Average DSC = ' + str(np.mean(DSC[r, :])) + ' .\n')
    output.close()
print 'DSC threshold = 0.90, ' + 'Average DSC = ' + str(np.mean(DSC_90)) + ' .'
print 'DSC threshold = 0.95, ' + 'Average DSC = ' + str(np.mean(DSC_95)) + ' .'
print 'DSC threshold = 0.98, ' + 'Average DSC = ' + str(np.mean(DSC_98)) + ' .'
print 'DSC threshold = 0.99, ' + 'Average DSC = ' + str(np.mean(DSC_99)) + ' .'
output = open(coarse2fine_result_file, 'a+')
output.write('DSC threshold = 0.90, ' + 'Average DSC = ' + str(np.mean(DSC_90)) + ' .\n')
output.write('DSC threshold = 0.95, ' + 'Average DSC = ' + str(np.mean(DSC_95)) + ' .\n')
output.write('DSC threshold = 0.98, ' + 'Average DSC = ' + str(np.mean(DSC_98)) + ' .\n')
output.write('DSC threshold = 0.99, ' + 'Average DSC = ' + str(np.mean(DSC_99)) + ' .\n')
output.close()
print 'The coarse-to-fine testing process is finished.'
