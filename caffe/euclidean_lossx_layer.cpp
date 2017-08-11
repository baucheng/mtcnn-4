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
  num_ = bottom[1]->num();
  channels_ = bottom[1]->channels();
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_set(count, (Dtype)0, diff_.mutable_cpu_data());
  if (has_ignore_label_) {
    const Dtype* b0 = bottom[0]->cpu_data();
    const Dtype* b1 = bottom[1]->cpu_data();
    Dtype* diff = diff_.mutable_cpu_data();
    count_valid_ = 0;
    for (int i = 0; i < num_; ++i) {
      int channels_ignore = 0;
      for (int c = 0; c < channels_; ++c) {
        if (b1[i * channels_ + c] == ignore_label_)
          ++channels_ignore;
      }
      if (channnels_ignore == channels__) {
        caffe_sub(channels_,
                  b0 + i * channels_,
                  b1 + i * channels_,
                  diff + i * channels_);
        ++count_valid_;
      }
    }
    Dtype loss = 0;
    Dtype dot = caffe_cpu_dot(count, diff, diff);
    if (count_valid_ > 0) {
      loss = dot / count_valid_ / Dtype(2);
    }
    top[0]->mutable_cpu_data()[0] = loss;
  } else {
    caffe_sub(
        count,
        bottom[0]->cpu_data(),
        bottom[1]->cpu_data(),
        diff_.mutable_cpu_data());
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (has_ignore_label_) {
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        if (count_valid_ > 0) {
          const Dtype sign = (i == 0) ? 1 : -1;
          const Dtype alpha = sign * top[0]->cpu_diff()[0] / count_valid_;
          caffe_cpu_axpby(
              bottom[i]->count(),                 // count
              alpha,                              // alpha
              diff_.cpu_data(),                   // a
              Dtype(0),                           // beta
              bottom[i]->mutable_cpu_diff());     // b
        } else {
          memset(bottom[i]->mutable_cpu_diff(), 0, sizeof(Dtype) * bottom[i]->count());
        }
      }
    }
  } else {
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
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossXLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossXLayer);
REGISTER_LAYER_CLASS(EuclideanLossX);

}  // namespace caffe
