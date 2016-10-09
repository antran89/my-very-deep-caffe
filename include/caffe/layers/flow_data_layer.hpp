/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef CAFFE_FLOW_DATA_LAYER_HPP
#define CAFFE_FLOW_DATA_LAYER_HPP

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

template <typename Dtype>
class FlowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit FlowDataLayer(const LayerParameter& param);
  virtual ~FlowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // FlowDataLayer uses FlowDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "FlowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  FlowDataReader reader_;

  // extract 10-view features
  int num_test_views_;
};

}  // namespace caffe

#endif // CAFFE_FLOW_DATA_LAYER_HPP

