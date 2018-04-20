import caffe
import numpy as np


class UncropLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) == 4:
            self.no_forward = True
        else:
            self.no_forward = False


    def reshape(self, bottom, top):
        top[0].reshape(*bottom[2].data.shape)


    def forward(self, bottom, top): 
        MIN_VALUE = -9999999.
        temp = bottom[1].data.astype(np.float32)
        crop_shape = bottom[0].data[0][0].astype(np.int16)
        original_shape = bottom[2].data.shape
        uncropped = np.zeros(original_shape).astype(np.float32)
        uncropped[...] = MIN_VALUE
        uncropped[:, :, crop_shape[0]: crop_shape[1], crop_shape[2]: crop_shape[3]] = temp
        top[0].data[...] = uncropped
        if self.no_forward == True and np.sum(bottom[3].data) == 0:
            top[0].data[...] = MIN_VALUE


    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff
        crop_shape = bottom[0].data[0][0].astype(np.int16)
        diff = diff[:, :, crop_shape[0]: crop_shape[1], crop_shape[2]: crop_shape[3]]
        bottom[1].diff[...] = diff
