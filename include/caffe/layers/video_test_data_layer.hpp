/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef CAFFE_VIDEO_TEST_DATA_LAYER_HPP_
#define CAFFE_VIDEO_TEST_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/flow_data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

// CAFFE: number of views when testing
const int CAFFE_NUM_TEST_VIEWS = 10;

template <typename Dtype>
class VideoTestDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoTestDataLayer(const LayerParameter& param);
  virtual ~VideoTestDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // VideoTestDataLayer uses FlowDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VideoTestData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  FlowDataReader reader_;
};

}  // namespace caffe

#endif // CAFFE_VIDEO_TEST_DATA_LAYER_HPP_

