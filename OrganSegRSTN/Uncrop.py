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
        if self.no_forward == True and np.sum(bottom[3].data) == 0:
            top[0].data[...] = MIN_VALUE
        else:
            top[0].data[...] = np.ones(bottom[2].data.shape, dtype = np.float32)
            top[0].data[...] *= (-9999999)
            shape_ = bottom[0].data[0][0].astype(np.int16)
            top[0].data[:, :, shape_[0]: shape_[1], shape_[2]: shape_[3]] = \
                bottom[1].data.astype(np.float32)


    def backward(self, top, propagate_down, bottom):
        shape_ = bottom[0].data[0][0].astype(np.int16)
        bottom[1].diff[...] = top[0].diff[:, :, shape_[0]: shape_[1], shape_[2]: shape_[3]]
