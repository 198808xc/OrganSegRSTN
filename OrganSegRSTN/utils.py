import numpy as np
import os
import sys
import math


####################################################################################################
# returning the binary label map by the organ ID (especially useful under overlapping cases)
#   label: the label matrix
#   organ_ID: the organ ID
def is_organ(label, organ_ID):
    return label == organ_ID


####################################################################################################
# determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]
def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - fold_remainder) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (i >= start_index and i < end_index)


####################################################################################################
# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')


####################################################################################################
# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')


####################################################################################################
# returning the filename of the log file
def log_filename(snapshot_directory):
    count = 0
    while True:
        count = count + 1
        if count == 1:
            log_file_ = os.path.join(snapshot_directory, 'log.txt')
        else:
            log_file_ = os.path.join(snapshot_directory, 'log' + str(count) + '.txt')
        if not os.path.isfile(log_file_):
            return log_file_


####################################################################################################
# determining if the loss values are reasonable (otherwise re-training is required)
def valid_loss(log_file, iterations):
    FRACTION = 0.02
    loss_avg = 0.0
    loss_min = 1.0
    count = 0
    text = open(log_file, 'r').read().splitlines()
    for l in range(int(len(text) - iterations / 5 * FRACTION - 10), len(text)):
        index1 = text[l].find('Iteration')
        index2 = text[l].find('(')
        index3 = text[l].find('loss = ')
        if index1 > 0 and index2 > index1 and index3 > index2:
            iteration = int(text[l][index1 + 10: index2 - 1])
            loss = float(text[l][index3 + 7: ])
            if iteration >= iterations * (1 - FRACTION):
                loss_avg += loss
                loss_min = min(loss_min, loss)
                count += 1
    if count > 0:
        loss_avg /= count
    else:
        loss_avg = loss
        loss_min = loss
    return loss_avg < 0.4 and loss_min < 0.35


####################################################################################################
# returning the snapshot filename according to the directory and the iteration count
def snapshot_filename(snapshot_directory, t):
    return os.path.join(snapshot_directory, 'train_iter_' + str(t) + '.caffemodel')


####################################################################################################
# returning the s-th latest timestamp that contains all the required snapshots
def snapshot_name_from_timestamp_s(snapshot_path, current_fold, \
    plane, stage_code, slice_thickness, organ_ID, iteration, timestamp, s):
    snapshot_prefix = 'FD' + str(current_fold) + ':' + plane + \
        stage_code + str(slice_thickness) + '_' + str(organ_ID) + '_'
    if len(timestamp) == 15:
        snapshot_prefix = snapshot_prefix + timestamp
    if not os.path.isdir(snapshot_path):
        return ''
    directory = os.listdir(snapshot_path)
    directory.sort()
    found = False
    count = 0
    for name in reversed(directory):
        if snapshot_prefix in name:
            snapshot_directory = os.path.join(snapshot_path, name)
            valid = True
            for t in range(len(iteration)):
                snapshot_file = snapshot_filename(snapshot_directory, iteration[t])
                if not os.path.isfile(snapshot_file):
                    valid = False
                    break
            if valid:
                count = count + 1
                if count == s:
                    snapshot_name = name
                    found = True
                    break
    if found:
        return snapshot_name
    else:
        return ''


####################################################################################################
# returning the latest timestamp that contains all the required snapshots
def snapshot_name_from_timestamp(snapshot_path, \
    current_fold, plane, stage_code, slice_thickness, organ_ID, iteration, timestamp):
    return snapshot_name_from_timestamp_s(snapshot_path, \
        current_fold, plane, stage_code, slice_thickness, organ_ID, iteration, timestamp, 1)


####################################################################################################
# returning the s-th latest timestamp that contains all the required snapshots (2-stage version)
def snapshot_name_from_timestamp_2_s(snapshot_path1, snapshot_path2, current_fold, plane, \
    stage_code1, stage_code2, slice_thickness, organ_ID, iteration, timestamp1, timestamp2, s):
    snapshot_prefix = 'FD' + str(current_fold) + ':'
    snapshot_str1 = plane + stage_code1 + str(slice_thickness) + '_' + str(organ_ID) + '_'
    if len(timestamp1) == 15:
        snapshot_str1 = snapshot_str1 + timestamp1
    snapshot_str2 = plane + stage_code2 + str(slice_thickness) + '_' + str(organ_ID) + '_'
    if len(timestamp2) == 15:
        snapshot_str2 = snapshot_str2 + timestamp2
    if not os.path.isdir(snapshot_path2):
        return ['', '']
    directory2 = os.listdir(snapshot_path2)
    directory2.sort()
    found = False
    count = 0
    for name2 in reversed(directory2):
        if snapshot_prefix in name2 and snapshot_str1 in name2 and snapshot_str2 in name2:
            name1 = name2.split(',')[0]
            snapshot_directory1 = os.path.join(snapshot_path1, name1)
            snapshot_directory2 = os.path.join(snapshot_path2, name2)
            print snapshot_str1, snapshot_str2, name2
            valid = True
            for t in range(len(iteration)):
                snapshot_file1 = snapshot_filename(snapshot_directory1, iteration[t])
                snapshot_file2 = snapshot_filename(snapshot_directory2, iteration[t])
                print snapshot_file1, snapshot_file2
                print os.path.isfile(snapshot_file1), os.path.isfile(snapshot_file2)
                if (os.path.isfile(snapshot_file1) and os.path.isfile(snapshot_file2)) or \
                    (not os.path.isfile(snapshot_file1) and not os.path.isfile(snapshot_file2)):
                    valid = False
                    break
            if valid:
                count = count + 1
                if count == s:
                    snapshot_name = [name1, name2]
                    found = True
                    break
    if found:
        return snapshot_name
    else:
        return ['', '']


####################################################################################################
# returning the latest timestamp that contains all the required snapshots (2-stage version)
def snapshot_name_from_timestamp_2(snapshot_path1, snapshot_path2, current_fold, plane, \
    stage_code1, stage_code2, slice_thickness, organ_ID, iteration, timestamp1, timestamp2):
    return snapshot_name_from_timestamp_2_s(snapshot_path1, snapshot_path2, current_fold, plane, \
        stage_code1, stage_code2, slice_thickness, organ_ID, iteration, timestamp1, timestamp2, 1)


####################################################################################################
# returning the volume filename as in the testing stage
def volume_filename_testing(result_directory, t, i):
    return os.path.join(result_directory, str(t) + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the volume filename as in the fusion stage
def volume_filename_fusion(result_directory, code, i):
    return os.path.join(result_directory, code + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the volume filename as in the coarse-to-fine testing stage
def volume_filename_coarse2fine(result_directory, r, i):
    return os.path.join(result_directory, 'R' + str(r) + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the s-th latest timestamp that contains all the required results
def result_name_from_timestamp_s(result_path, current_fold, \
    plane, stage_code, slice_thickness, organ_ID, iteration, volume_list, timestamp, s):
    result_prefix = 'FD' + str(current_fold) + ':' + plane + \
        stage_code + str(slice_thickness) + '_' + str(organ_ID) + '_'
    if len(timestamp) == 15:
        result_prefix = result_prefix + timestamp
    if not os.path.isdir(result_path):
        return ''
    directory = os.listdir(result_path)
    directory.sort()
    found = False
    count = 0
    for name in reversed(directory):
        if result_prefix in name and not name.endswith('_'):
            result_directory = os.path.join(result_path, name, 'volumes')
            valid = True
            for t in range(len(iteration)):
                for i in range(len(volume_list)):
                    volume_file = volume_filename_testing(result_directory, iteration[t], i)
                    if not os.path.isfile(volume_file):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                count = count + 1
                if count == s:
                    result_name = name
                    found = True
                    break
    if found:
        return result_name
    else:
        return ''


####################################################################################################
# returning the latest timestamp that contains all the required results
def result_name_from_timestamp(result_path, current_fold, \
    plane, stage_code, slice_thickness, organ_ID, iteration, volume_list, timestamp):
    return result_name_from_timestamp_s(result_path, current_fold, \
        plane, stage_code, slice_thickness, organ_ID, iteration, volume_list, timestamp, 1)


####################################################################################################
# returning the s-th latest timestamp that contains all the required results (2-stage version)
def result_name_from_timestamp_2_s(result_path, \
    current_fold, plane, stage_code1, stage_code2, slice_thickness, organ_ID, \
    iteration, volume_list, timestamp1, timestamp2, s):
    result_prefix = 'FD' + str(current_fold) + ':'
    result_str1 = plane + stage_code1 + str(slice_thickness) + '_' + str(organ_ID) + '_'
    if len(timestamp1) == 15:
        result_str1 = result_str1 + timestamp1
    result_str2 = plane + stage_code2 + str(slice_thickness) + '_' + str(organ_ID) + '_'
    if len(timestamp2) == 15:
        result_str2 = result_str2 + timestamp2
    if not os.path.isdir(result_path):
        return ''
    directory = os.listdir(result_path)
    directory.sort()
    found = False
    count = 0
    for name in reversed(directory):
        if result_prefix in name and result_str1 in name and result_str2 in name:
            result_directory = os.path.join(result_path, name, 'volumes')
            valid = True
            for t in range(len(iteration)):
                for i in range(len(volume_list)):
                    volume_file = volume_filename_testing(result_directory, iteration[t], i)
                    if not os.path.isfile(volume_file):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                count = count + 1
                if count == s:
                    result_name = name
                    found = True
                    break
    if found:
        return result_name
    else:
        return ''


####################################################################################################
# returning the latest timestamp that contains all the required results (2-stage version)
def result_name_from_timestamp_2(result_path, \
    current_fold, plane, stage_code1, stage_code2, slice_thickness, organ_ID, \
    iteration, volume_list, timestamp1, timestamp2):
    return result_name_from_timestamp_2_s(result_path, \
        current_fold, plane, stage_code1, stage_code2, slice_thickness, organ_ID, \
        iteration, volume_list, timestamp1, timestamp2, 1)


####################################################################################################
# computing the DSC together with other values based on the label and prediction volumes
def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.logical_and(pred, label).sum()
    return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum


####################################################################################################
# post-processing: preserving the largest connecting component(s) and discarding other voxels
#     The floodfill algorithm is used to detect the connecting components.
#     In the future version, this function is to be replaced by a C module for speedup!
#   F: a binary volume, the volume to be post-processed
#   S: a binary volume, the seed voxels (currently defined as those predicted as FG by all 3 views)
#       NOTE: a connected component will not be considered if it does not contain any seed voxels
#   threshold: a floating point number in [0, 1] determining if a connected component is accepted
#       NOTE: accepted if it is not smaller larger than the largest volume times this number
#       NOTE: 1 means to only keep the largest one(s), 0 means to keep all
#   organ_ID: passed in case that each organ needs to be dealt with differently
def post_processing(F, S, threshold, organ_ID):
    if F.sum() == 0:
        return F
    if F.sum() >= np.product(F.shape) / 2:
        return F
    height  = F.shape[0]
    width = F.shape[1]
    depth = F.shape[2]
    ll = np.array(np.nonzero(S))
    marked = np.zeros_like(F, dtype = np.bool)
    queue = np.zeros((F.sum(), 3), dtype = np.int)
    volume = np.zeros(F.sum(), dtype = np.int)
    head = 0
    tail = 0
    bestHead = 0
    bestTail = 0
    bestHead2 = 0
    bestTail2 = 0
    for l in range(ll.shape[1]):
        if not marked[ll[0, l], ll[1, l], ll[2, l]]:
            temp = head
            marked[ll[0, l], ll[1, l], ll[2, l]] = True
            queue[tail, :] = [ll[0, l], ll[1, l], ll[2, l]]
            tail = tail + 1
            while (head < tail):
                t1 = queue[head, 0]
                t2 = queue[head, 1]
                t3 = queue[head, 2]
                if t1 > 0 and F[t1 - 1, t2, t3] and not marked[t1 - 1, t2, t3]:
                    marked[t1 - 1, t2, t3] = True
                    queue[tail, :] = [t1 - 1, t2, t3]
                    tail = tail + 1
                if t1 < height - 1 and F[t1 + 1, t2, t3] and not marked[t1 + 1, t2, t3]:
                    marked[t1 + 1, t2, t3] = True
                    queue[tail, :] = [t1 + 1, t2, t3]
                    tail = tail + 1
                if t2 > 0 and F[t1, t2 - 1, t3] and not marked[t1, t2 - 1, t3]:
                    marked[t1, t2 - 1, t3] = True
                    queue[tail, :] = [t1, t2 - 1, t3]
                    tail = tail + 1
                if t2 < width - 1 and F[t1, t2 + 1, t3] and not marked[t1, t2 + 1, t3]:
                    marked[t1, t2 + 1, t3] = True
                    queue[tail, :] = [t1, t2 + 1, t3]
                    tail = tail + 1
                if t3 > 0 and F[t1, t2, t3 - 1] and not marked[t1, t2, t3 - 1]:
                    marked[t1, t2, t3 - 1] = True
                    queue[tail, :] = [t1, t2, t3 - 1]
                    tail = tail + 1
                if t3 < depth - 1 and F[t1, t2, t3 + 1] and not marked[t1, t2, t3 + 1]:
                    marked[t1, t2, t3 + 1] = True
                    queue[tail, :] = [t1, t2, t3 + 1]
                    tail = tail + 1
                head = head + 1
            if tail - temp > bestTail - bestHead:
                bestHead2 = bestHead
                bestTail2 = bestTail
                bestHead = temp
                bestTail = tail
            elif tail - temp > bestTail2 - bestHead2:
                bestHead2 = temp
                bestTail2 = tail
            volume[temp: tail] = tail - temp
    volume = volume[0: tail]
    target_voxel = np.where(volume >= (bestTail - bestHead) * threshold)
    F0 = np.zeros_like(F, dtype = np.bool)
    F0[tuple(map(tuple, np.transpose(queue[target_voxel, :])))] = True
    return F0


####################################################################################################
# defining the common variables used throughout the entire flowchart
data_path = sys.argv[1]
image_path = os.path.join(data_path, 'images')
image_path_ = {}
for plane in ['X', 'Y', 'Z']:
    image_path_[plane] = os.path.join(data_path, 'images_' + plane)
    if not os.path.exists(image_path_[plane]):
        os.makedirs(image_path_[plane])
label_path = os.path.join(data_path, 'labels')
label_path_ = {}
for plane in ['X', 'Y', 'Z']:
    label_path_[plane] = os.path.join(data_path, 'labels_' + plane)
    if not os.path.exists(label_path_[plane]):
        os.makedirs(label_path_[plane])
list_path = os.path.join(data_path, 'lists')
if not os.path.exists(list_path):
    os.makedirs(list_path)
list_training = {}
for plane in ['X', 'Y', 'Z']:
    list_training[plane] = os.path.join(list_path, 'training_' + plane + '.txt')
CAFFE_root = os.path.join(data_path, 'libs', 'caffe-master');
prototxt_path = os.path.join(data_path, 'prototxts')
if not os.path.exists(prototxt_path):
    os.makedirs(prototxt_path)
model_path = os.path.join(data_path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)
pretrained_model_path = os.path.join(data_path, 'models', 'pretrained')
if not os.path.exists(pretrained_model_path):
    os.makedirs(pretrained_model_path)
snapshot_path = os.path.join(data_path, 'models', 'snapshots')
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
log_path = os.path.join(data_path, 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)
result_path = os.path.join(data_path, 'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)
