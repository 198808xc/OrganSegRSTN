import caffe
import numpy as np
import random


class CropLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.margin = 0
        self.prob = 0
        self.batch = 0
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
        params = eval(self.param_str)
        self.TEST = params["TEST"]


    def reshape(self, bottom, top):
        (N, C, W, H) = bottom[0].data.shape
        data = bottom[1].data
        binary_mask = (bottom[0].data >= 0.5).astype(np.uint8)
        if len(bottom) == 6 and np.sum(binary_mask) == 0:
            binary_mask = (bottom[5].data >= 0.5).astype(np.uint8)
        self.margin = int(bottom[2].data)
        self.prob = float(bottom[3].data)
        self.batch = int(bottom[4].data)
        if self.TEST == 1:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            self.update_margin()
        if np.sum(binary_mask) == 0:
            minA = 0
            maxA = W
            minB = 0
            maxB = H
            self.no_forward = True
        else:
            if N > 1:
                mask = np.zeros(shape = (N, C, W, H))
                for n in range(N):
                    cur_mask = binary_mask[n, :, :, :]
                    arr = np.nonzero(cur_mask)
                    minA = min(arr[1])
                    maxA = max(arr[1])
                    minB = min(arr[2])
                    maxB = max(arr[2])
                    bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
			int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
                    mask[n, :, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
                data = data * mask

            arr = np.nonzero(binary_mask)
            minA = min(arr[2])
            maxA = max(arr[2])
            minB = min(arr[3])
            maxB = max(arr[3])
            self.no_forward = False
        self.bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
            int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
        self.cropped_image = data[:, :, self.bbox[0]: self.bbox[1], \
            self.bbox[2]: self.bbox[3]].copy().astype(np.float32)
        top[0].reshape(*self.cropped_image.shape)
        top[1].reshape(1, 2, 4)


    def forward(self, bottom, top): 
        if self.no_forward == True and self.TEST == 1:
            top[0].data[...] = 0.
        else:
            top[0].data[...] = self.cropped_image
        top[1].data[...] = np.zeros((1, 2, 4), dtype = np.int16)
        top[1].data[0][0] = self.bbox
        top[1].data[0][1] = bottom[0].data.shape


    def backward(self, top, propagate_down, bottom):
        diff = np.zeros(bottom[0].data.shape)
        diff[:, :, self.bbox[0]: self.bbox[1], self.bbox[2]: self.bbox[3]] = top[0].diff
        bottom[1].diff[...] = diff


    def update_margin(self):
        MAX_INT = 256
        if random.randint(0, MAX_INT - 1) >= MAX_INT * self.prob:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            a = np.zeros(self.batch * 4, dtype = np.uint8)
            for i in range(self.batch * 4):
                a[i] = random.randint(0, self.margin * 2)
            self.left = int(a[0: self.batch].sum() / self.batch)
            self.right = int(a[self.batch: self.batch * 2].sum() / self.batch)
            self.top = int(a[self.batch * 2: self.batch * 3].sum() / self.batch)
            self.bottom = int(a[self.batch * 3: self.batch * 4].sum() / self.batch)
