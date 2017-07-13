#ifndef CAFFE_VIDEO_CLIP_DATA_LAYER_HPP_
#define CAFFE_VIDEO_CLIP_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_clip_data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class VideoClipDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoClipDataLayer(const LayerParameter& param);
  virtual ~VideoClipDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // VideoClipDataLayer uses VideoClipDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VideoClipData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  VideoClipDataReader reader_;

  // extract 10-view features
  int num_test_views_;
};

}  // namespace caffe

#endif  // CAFFE_VIDEO_CLIP_DATA_LAYER_HPP_
