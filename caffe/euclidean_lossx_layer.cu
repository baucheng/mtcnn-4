#include <vector>
#include <iostream>

#include "caffe/layers/euclidean_lossx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EuclideanLossForwardGPU(const int nthreads,
          Dtype* diff, const Dtype* b0, const Dtype* b1, Dtype* loss,
          const int channels, const bool has_ignore_label,
          const int ignore_label, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //const int n = index / channels;
    //const int c = index % channels;
    const int label_value = static_cast<int>(b1[index]);
    if (has_ignore_label && label_value == ignore_label) {
      diff[index] = 0;
      loss[index] = 0;
      counts[index] = 0;
    } else {
      diff[index] = b0[index] - b1[index];
      loss[index] = (b0[index] - b1[index]) * (b0[index] - b1[index]);
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_set(count, (Dtype)0, diff_.mutable_gpu_data());
  if (has_ignore_label_) {
    const Dtype* b0 = bottom[0]->gpu_data();
    const Dtype* b1 = bottom[1]->gpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = diff_.mutable_gpu_data();
    //std::cout << diff_.shape_string() << std::endl;
    const int nthreads = num_ * channels_;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    Dtype* counts = bottom[1]->mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    EuclideanLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, diff, b0, b1, loss_data,
        channels_, has_ignore_label_, ignore_label_, counts);
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype count_valid = 0;
    if (has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &count_valid);
    }
    count_valid_ = count_valid / channels_;
    if (count_valid_ > 0)
      loss = loss / count_valid_ / Dtype(2);
      //std::cout << "count_valid_ " << count_valid_ << std::endl;
    top[0]->mutable_cpu_data()[0] = loss;
  } else {
    caffe_gpu_sub(count,
                  bottom[0]->gpu_data(),
                  bottom[1]->gpu_data(),
                  diff_.mutable_gpu_data());
    Dtype dot;
    caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  } 
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (has_ignore_label_) {
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        if (count_valid_ > 0) {
          const Dtype sign = (i == 0) ? 1 : -1;
          const Dtype alpha = sign * top[0]->cpu_diff()[0] / count_valid_;
          caffe_gpu_axpby(
              bottom[i]->count(),                 // count
              alpha,                              // alpha
              diff_.gpu_data(),                   // a
              Dtype(0),                           // beta
              bottom[i]->mutable_gpu_diff());     // b
        } else {
          caffe_gpu_memset(sizeof(Dtype)*bottom[i]->count(), 0, bottom[i]->mutable_gpu_diff());
        }
      }
    }
  } else {
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        caffe_gpu_axpby(
            bottom[i]->count(),                 // count
            alpha,                              // alpha
            diff_.gpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[i]->mutable_gpu_diff());     // b
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossXLayer);

}  // namespace caffe
