/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/twostream_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
TwostreamDataLayer<Dtype>::TwostreamDataLayer(const LayerParameter& param)
    : BasePrefetchingTwostreamDataLayer<Dtype>(param),
      reader_(param), num_test_views_(1) {
}

template <typename Dtype>
TwostreamDataLayer<Dtype>::~TwostreamDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void TwostreamDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
    const int batch_size = this->layer_param_.twostream_data_param().batch_size();
    num_test_views_ = this->layer_param_.twostream_data_param().test_10view_features() ? 10 : 1;
    if (num_test_views_ == 10)
        CHECK_EQ(this->phase_, TEST) << "Extracting 10-view features is only available in TEST phase";
    if (this->phase_ == TEST)
        LOG(INFO) << "Extracting " << num_test_views_ << "-view features in TEST phase.";

    // Read a rgb data point, and use it to initialize the first top blob.
    Datum& rgb_datum = *(reader_.rgb_full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(rgb_datum);
    top_shape[0] = num_test_views_;
    this->transformed_rgb_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size * num_test_views_;
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].rgb_data_.Reshape(top_shape);
    }
    LOG(INFO) << "rgb data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();

    // Read a flow data point, and use it to initialize the second top blob.
    Datum& flow_datum = *(reader_.flow_full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    top_shape = this->data_transformer_->InferBlobShape(flow_datum);
    top_shape[0] = num_test_views_;
    this->transformed_flow_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size * num_test_views_;
    top[1]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].flow_data_.Reshape(top_shape);
    }
    LOG(INFO) << "flow data size: " << top[1]->num() << ","
              << top[1]->channels() << "," << top[1]->height() << ","
              << top[1]->width();

    // label
    if (this->output_labels_) {
        vector<int> label_shape(1, batch_size);
        top[2]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }
    }
}

// This function is called on prefetch thread
template<typename Dtype>
void TwostreamDataLayer<Dtype>::load_batch(TwostreamBatch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->rgb_data_.count());
    CHECK(this->transformed_rgb_data_.count());
    CHECK(batch->flow_data_.count());
    CHECK(this->transformed_flow_data_.count());

    // Reshape according to the first datum of each batch
    // on single input batches allows for inputs of varying dimension.
    const int batch_size = this->layer_param_.twostream_data_param().batch_size();
    // rgb datum reshape
    Datum& rgb_datum = *(reader_.rgb_full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(rgb_datum);
    top_shape[0] = num_test_views_;
    this->transformed_rgb_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size * num_test_views_;
    batch->rgb_data_.Reshape(top_shape);

    // flow datum reshape
    Datum& flow_datum = *(reader_.flow_full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    top_shape = this->data_transformer_->InferBlobShape(flow_datum);
    top_shape[0] = num_test_views_;
    this->transformed_flow_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size * num_test_views_;
    batch->flow_data_.Reshape(top_shape);

    Dtype* top_rgb_data = batch->rgb_data_.mutable_cpu_data();
    Dtype* top_flow_data = batch->flow_data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

    if (this->output_labels_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a datum
        Datum& rgb_datum = *(reader_.rgb_full().pop("Waiting for rgb data"));
        Datum& flow_datum = *(reader_.flow_full().pop("Waiting for flow data"));
        read_time += timer.MicroSeconds();
        //    DLOG(INFO) << "number of data in full queue: " << reader_.full().size();
        timer.Start();
        // Apply data transformations (mirror, scale, crop...) to rgb data
        int offset = batch->rgb_data_.offset(item_id * num_test_views_);
        this->transformed_rgb_data_.set_cpu_data(top_rgb_data + offset);
        // Copy label.
        if (this->output_labels_) {
            top_label[item_id] = rgb_datum.label();
        }
        // Apply data transformations (mirror, scale, crop...) to rgb data
        offset = batch->flow_data_.offset(item_id * num_test_views_);
        this->transformed_flow_data_.set_cpu_data(top_flow_data + offset);
        if (this->phase_ == TRAIN)
            this->data_transformer_->TransformVariedSizeTwostreamDatum(rgb_datum, flow_datum,
                                                                       &(this->transformed_rgb_data_), &(this->transformed_flow_data_));
        else if (this->phase_ == TEST)
            this->data_transformer_->TransformVariedSizeTwostreamTestDatum(rgb_datum, flow_datum,
                                                                           &(this->transformed_rgb_data_), &(this->transformed_flow_data_), num_test_views_);

        trans_time += timer.MicroSeconds();

        // push processed datum back into free queue
        reader_.rgb_free().push(const_cast<Datum*>(&rgb_datum));
        reader_.flow_free().push(const_cast<Datum*>(&flow_datum));
    }
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TwostreamDataLayer);
REGISTER_LAYER_CLASS(TwostreamData);

}  // namespace caffe
