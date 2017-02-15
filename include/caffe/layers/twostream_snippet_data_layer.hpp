/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef CAFFE_TWOSTREAM_SNIPPET_DATA_LAYER_HPP_
#define CAFFE_TWOSTREAM_SNIPPET_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/twostream_snippet_data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_twostream_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class TwostreamSnippetDataLayer : public BasePrefetchingTwostreamDataLayer<Dtype> {
 public:
  explicit TwostreamSnippetDataLayer(const LayerParameter& param);
  virtual ~TwostreamSnippetDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // FlowDataLayer uses FlowDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "TwostreamSnippetData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void load_batch(TwostreamBatch<Dtype>* batch);

  TwostreamSnippetDataReader reader_;

  // blob to store transformed data
  Blob<Dtype> transformed_rgb_data_;
  Blob<Dtype> transformed_flow_data_;

  // extract 10-view features
  int num_test_views_;
};

}  // namespace caffe

#endif // CAFFE_TWOSTREAM_SNIPPET_DATA_LAYER_HPP_

