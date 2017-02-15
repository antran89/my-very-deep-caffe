/*
 * Copyright (C) 2017 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/video_snippet_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
VideoSnippetDataLayer<Dtype>::VideoSnippetDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param), num_test_views_(1) {
}

template <typename Dtype>
VideoSnippetDataLayer<Dtype>::~VideoSnippetDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoSnippetDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.video_snippet_data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());
  num_test_views_ = this->layer_param_.video_snippet_data_param().test_10view_features() ? 10 : 1;
  if (num_test_views_ == 10)
      CHECK_EQ(this->phase_, TEST) << "Extracting 10-view features is only available in TEST phase";
  if (this->phase_ == TEST)
      LOG(INFO) << "Extracting " << num_test_views_ << "-view features in TEST phase.";

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  top_shape[0] = num_test_views_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size * num_test_views_;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void VideoSnippetDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.video_snippet_data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  top_shape[0] = num_test_views_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size * num_test_views_;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for flow data"));
    read_time += timer.MicroSeconds();
//    DLOG(INFO) << "number of data in full queue: " << reader_.full().size();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id * num_test_views_);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (this->phase_ == TRAIN)
        this->data_transformer_->TransformVariedSizeDatum(datum, &(this->transformed_data_));
    else if (this->phase_ == TEST)
        this->data_transformer_->TransformVariedSizeTestDatum(datum, &(this->transformed_data_), num_test_views_);

    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoSnippetDataLayer);
REGISTER_LAYER_CLASS(VideoSnippetData);

}  // namespace caffe
