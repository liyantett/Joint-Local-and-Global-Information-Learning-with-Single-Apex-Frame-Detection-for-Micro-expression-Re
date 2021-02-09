#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  for (int i = 1; i < bottom.size(); ++i) {
    const Dtype* bottom_data_i = bottom[i]->gpu_data();
    caffe_gpu_axpby(bottom[0]->count(), Dtype(1.0), bottom_data_i,
       Dtype(1.0), top_data); 
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i=0; i<bottom.size(); i++){
    if(propagate_down[i])
    {
      const Dtype* top_diff=top[0]->gpu_diff();
      Dtype* bottom_diff=bottom[i]->mutable_gpu_diff();
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }

  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SumLayer);

}  // namespace caffe
