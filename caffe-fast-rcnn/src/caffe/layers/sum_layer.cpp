#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->count(), bottom[0]->count()) <<" Blobs must be same size."; 
    CHECK_EQ(bottom[i]->width(), bottom[0]->width()) <<" Blobs must be same size.";
    CHECK_EQ(bottom[i]->height(), bottom[0]->height()) <<" Blobs must be same size.";
    CHECK_EQ(bottom[i]->channels(), bottom[0]->channels()) <<" Blobs must be same size.";
    
    
    
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  for (int i = 1; i < bottom.size(); ++i) {
    const Dtype* bottom_data_i = bottom[i]->cpu_data();
    caffe_cpu_axpby(bottom[0]->count(), Dtype(1.0), bottom_data_i,
       Dtype(1.0), top_data); 
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i=0; i<bottom.size(); i++){
    if(propagate_down[i])
    {
      const Dtype* top_diff=top[0]->cpu_diff();
      Dtype* bottom_diff=bottom[i]->mutable_cpu_diff();
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }

  }
}


#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(Sum);

}  // namespace caffe
