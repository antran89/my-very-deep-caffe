/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef CAFFE_TWOSTREAM_DATA_LAYER_HPP
#define CAFFE_TWOSTREAM_DATA_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/twostream_data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_twostream_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class TwostreamDataLayer : public BasePrefetchingTwostreamDataLayer<Dtype> {
 public:
  explicit TwostreamDataLayer(const LayerParameter& param);
  virtual ~TwostreamDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // FlowDataLayer uses FlowDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "TwostreamData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  TwostreamDataReader reader_;

  // blob to store transformed data
  Blob<Dtype> transformed_rgb_data_;
  Blob<Dtype> transformed_flow_data_;
};

}  // namespace caffe

#endif // CAFFE_TWOSTREAM_DATA_LAYER_HPP

