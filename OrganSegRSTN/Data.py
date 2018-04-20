import numpy as np
import sys
import math
import caffe
import random
from PIL import Image
from joint_training import *


sys.path.insert(0, CAFFE_root + 'python')


class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.random = True
        self.seed = 1337
        image_list = open(training_set_filename(current_fold), 'r').read().splitlines()
        self.training_image_set = np.zeros((len(image_list)), dtype = np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.training_image_set[i] = int(s[0])
        slice_list = open(list_training[plane], 'r').read().splitlines()
        self.slices = len(slice_list)
        self.image_ID = np.zeros((self.slices), dtype = np.int)
        self.slice_ID = np.zeros((self.slices), dtype = np.int)
        self.image_filename = ['' for l in range(self.slices)]
        self.label_filename = ['' for l in range(self.slices)]
        self.average = np.zeros((self.slices))
        self.pixels = np.zeros((self.slices), dtype = np.int)
        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            self.slice_ID[l] = s[1]
            self.image_filename[l] = s[2]
            self.label_filename[l] = s[3]
            self.average[l] = float(s[4])
            self.pixels[l] = int(s[organ_ID * 5])
        if slice_threshold <= 1:
            pixels_index = sorted(range(self.slices), key = lambda l: self.pixels[l])
            last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))
            min_pixels = self.pixels[pixels_index[-last_index]]
        else:
            min_pixels = slice_threshold
        self.active_index = [l for l, p in enumerate(self.pixels) if p >= min_pixels]
        self.index_ = -1
        if self.random:
            random.seed(self.seed)
        self.next_slice_index()


    def next_slice_index(self):
        while True:
            if self.random:
                self.index_ = random.randint(0, len(self.active_index) - 1)
            else:
                self.index_ += 1
                if self.index_ == len(self.active_index):
                    self.index_ = 0
            self.index1 = self.active_index[self.index_]
            if self.image_ID[self.index1] in self.training_image_set:
                break
        self.index0 = self.index1 - 1
        if self.index1 == 0 or self.slice_ID[self.index0] <> self.slice_ID[self.index1] - 1:
            self.index0 = self.index1
        self.index2 = self.index1 + 1
        if self.index1 == self.slices - 1 or \
            self.slice_ID[self.index2] <> self.slice_ID[self.index1] + 1:
            self.index2 = self.index1


    def reshape(self, bottom, top):
        self.data, self.label = self.load_data()
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape((1))
        top[3].reshape((1))
        top[4].reshape((1))


    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = margin
        top[3].data[...] = prob
        top[4].data[...] = sample_batch
        self.next_slice_index()


    def backward(self, top, propagate_down, bottom):
        pass


    def load_data(self):
        if slice_thickness == 1:
            image1 = np.load(self.image_filename[self.index1])
            label1 = np.load(self.label_filename[self.index1])
            width = label1.shape[0]
            height = label1.shape[1]
            image = np.repeat(image1.reshape(1, width, height), 3, axis = 0)
            label = label1.reshape(1, width, height)
        elif slice_thickness == 3:
            image0 = np.load(self.image_filename[self.index0])
            image1 = np.load(self.image_filename[self.index1])
            image2 = np.load(self.image_filename[self.index2])
            label0 = np.load(self.label_filename[self.index0])
            label1 = np.load(self.label_filename[self.index1])
            label2 = np.load(self.label_filename[self.index2])
            width = label1.shape[0]
            height = label1.shape[1]
            image = np.concatenate((image0.reshape(1, width, height), \
                image1.reshape(1, width, height), image2.reshape(1, width, height)))
            label = np.concatenate((label0.reshape(1, width, height), \
                label1.reshape(1, width, height), label2.reshape(1, width, height)))
        image = image.astype(np.float32)
        image[image < low_range] = low_range
        image[image > high_range] = high_range
        image = (image - low_range) / (high_range - low_range)
        label = is_organ(label, organ_ID).astype(np.uint8)
        return image, label
