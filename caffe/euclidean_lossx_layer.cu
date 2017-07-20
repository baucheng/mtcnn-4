#include <vector>

#include "caffe/layers/euclidean_lossx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  int count = bottom[0]->count();
  caffe_gpu_set(count, (Dtype)0, diff_.mutable_gpu_data());
  if (has_ignore_label_) {
    const Dtype* output = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = diff_.mutable_gpu_data();
    int num = 0;
    for (int i = 0; i < outer_num_; ++i) {
      // we assume unified(ignore) label for single sample,
      // so just check the first element.
      const int label_value = static_cast<int>(label[i * inner_num_]);
      if (label_value != ignore_label_) {
        caffe_gpu_sub(inner_num_,
                      output + i * inner_num_,
                      label + i * inner_num_,
                      diff + i * inner_num_);                    
        Dtype dot;
        caffe_gpu_dot(inner_num_, diff + i * inner_num_,
                      diff + i * inner_num_, &dot);
        loss += dot;
        ++num;
      }
    }
    if (0 == num)
      ++num;
    loss = loss / num / Dtype(2);
  } else {
    caffe_gpu_sub(count,
                  bottom[0]->gpu_data(),
                  bottom[1]->gpu_data(),
                  diff_.mutable_gpu_data());
    Dtype dot;
    caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    loss = dot / bottom[0]->num() / Dtype(2);
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossXLayer);

}  // namespace caffe
