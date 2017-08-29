#include <vector>

#include "caffe/layers/euclidean_lossx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EuclideanLossXForwardGPU(const int nthreads,
          Dtype* diff, const Dtype* b0, const Dtype* b1, Dtype* loss,
          const bool has_ignore_label, const int ignore_label, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(b1[index]);
    if (has_ignore_label && label_value == ignore_label) {
      diff[index] = 0;
      loss[index] = 0;
      counts[index] = 0;
    } else {
      diff[index] = b0[index] - b1[index];
      loss[index] = 0.5 * (b0[index] - b1[index]) * (b0[index] - b1[index]);
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (!has_ignore_label_) {
    LOG(FATAL) << "Use EuclideanLossLayer instead.";
  }
  int count = bottom[0]->count();
  const Dtype* b0 = bottom[0]->gpu_data();
  const Dtype* b1 = bottom[1]->gpu_data();
  Dtype* diff = diff_.mutable_gpu_data();
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* counts = bottom[1]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  EuclideanLossXForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, diff, b0, b1, loss_data,
      has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalization_ == LossParameter_NormalizationMode_VALID) {
    caffe_gpu_asum(nthreads, counts, &valid_count_);
  }
  top[0]->mutable_cpu_data()[0] = loss; 
}

template <typename Dtype>
void EuclideanLossXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!has_ignore_label_) {
    LOG(FATAL) << "Use EuclideanLossLayer instead.";
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];
      caffe_gpu_axpby(
          bottom[i]->count(),                 // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());     // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossXLayer);

}  // namespace caffe
