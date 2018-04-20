#include <vector>
#include <fstream>
#include <iostream>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define FLT_EPSILON 0.000001

namespace caffe {

template <typename Dtype>
void DiceLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "DICE_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const int count = bottom[0]->count();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 1;
  Dtype up = (Dtype) FLT_EPSILON;
  Dtype down = (Dtype) FLT_EPSILON;
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  for (int i = 0; i < count; ++i) {
    up += 2 * target[i] * sigmoid_output_data[i];
    down += target[i] + sigmoid_output_data[i];
  }
  loss -= up / down;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype intersection = 0;
    Dtype Union = 0;
    for (int i = 0; i < count; ++i) {
      intersection +=  target[i] * sigmoid_output_data[i];
      Union += target[i] + sigmoid_output_data[i];
    }
    Dtype down = (Union + (Dtype) FLT_EPSILON) * (Union + (Dtype) FLT_EPSILON);
    for (int i = 0; i < count; ++i) {
      Dtype up = 2 * target[i] * (Union + (Dtype) FLT_EPSILON) -
        2 * intersection - (Dtype) FLT_EPSILON;
      bottom_diff[i] = - (up / down) * sigmoid_output_data[i] *
        (1 - sigmoid_output_data[i]);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
// STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(DiceLossLayer);
REGISTER_LAYER_CLASS(DiceLoss);

}  // namespace caffe
