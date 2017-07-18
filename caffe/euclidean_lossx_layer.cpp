#include <vector>

#include "caffe/layers/euclidean_lossx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  label_axis_ = 1; // default 1, not configurable now.
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  int count = bottom[0]->count();
  caffe_set(count, (Dtype)0, diff_.mutable_cpu_data());
  if (has_ignore_label_) {
    const Dtype* output = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = diff_.mutable_cpu_data();
    int num = 0;
    for (int i = 0; i < outer_num_; ++i) {
      // we assume unified(ignore) label for single sample,
      // so just check the first element.
      const int label_value = static_cast<int>(label[i * inner_num_]);
      if (label_value == ignore_label_)
        continue;
      caffe_sub(inner_num_,
                output + i * inner_num_,
                label + i * inner_num_,
                diff + i * inner_num_);
      loss += caffe_cpu_dot(inner_num_, diff + i * inner_num_,
                            diff + i * inner_num_);
      ++num;
    }
    if (0 == num)
      ++num;
    loss = loss / num / Dtype(2);
  } else {
    caffe_sub(
        count,
        bottom[0]->cpu_data(),
        bottom[1]->cpu_data(),
        diff_.mutable_cpu_data());
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    loss = dot / bottom[0]->num() / Dtype(2);
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossXLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossXLayer);
REGISTER_LAYER_CLASS(EuclideanLossX);

}  // namespace caffe
