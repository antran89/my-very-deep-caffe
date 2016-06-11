#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe{

template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
    this->StopInternalThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    const int new_height  = this->layer_param_.video_data_param().new_height();
    const int new_width  = this->layer_param_.video_data_param().new_width();
    const int new_length  = this->layer_param_.video_data_param().new_length();
    const int num_segments = this->layer_param_.video_data_param().num_segments();
    const string& source = this->layer_param_.video_data_param().source();

    LOG(INFO) << "Opening file: " << source;
    std:: ifstream infile(source.c_str());
    string filename;
    int label;
    int length;
    while (infile >> filename >> length >> label){
        lines_.push_back(std::make_pair(filename,label));
        lines_duration_.push_back(length);
    }
    if (this->layer_param_.video_data_param().shuffle()){
        const unsigned int prefectch_rng_seed = caffe_rng_rand();
        prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
        prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
        ShuffleVideos();
    }

    LOG(INFO) << "A total of " << lines_.size() << " videos.";
    lines_id_ = 0;

    Datum datum;
    const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
    frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
    int average_duration = (int) lines_duration_[lines_id_]/num_segments;
    vector<int> offsets;
    for (int i = 0; i < num_segments; ++i){
        caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
        int offset = (*frame_rng)() % (average_duration - new_length + 1);
        offsets.push_back(offset+i*average_duration);
    }
    if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
        CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum));
    else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FOREGROUND_SALIENCY)
        CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, false));
    else
        CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true));
    const int crop_size = this->layer_param_.transform_param().crop_size();
    const int batch_size = this->layer_param_.video_data_param().batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    if (crop_size > 0){
        top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
        for (int i = 0; i < this->PREFETCH_COUNT; i++)
            this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
    } else {
        top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        for (int i = 0; i < this->PREFETCH_COUNT; i++)
            this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
    }
    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

    // label
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; i++)
        this->prefetch_[i].label_.Reshape(label_shape);

    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos() {
    caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
    caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
    shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    Datum datum;
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    VideoDataParameter video_data_param = this->layer_param_.video_data_param();
    const int batch_size = video_data_param.batch_size();
    const int new_height = video_data_param.new_height();
    const int new_width = video_data_param.new_width();
    const int new_length = video_data_param.new_length();
    const int num_segments = video_data_param.num_segments();
    const int lines_size = lines_.size();

    // do we need to reshape batcha data before deferencing the pointer? NO
    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    for (int item_id = 0; item_id < batch_size; ++item_id){
        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);
        vector<int> offsets;
        int average_duration = (int) lines_duration_[lines_id_] / num_segments;
        for (int i = 0; i < num_segments; ++i) {
            if (this->phase_==TRAIN){
                caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
                int offset = (*frame_rng)() % (average_duration - new_length + 1);
                offsets.push_back(offset+i*average_duration);
            } else{
                offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
            }
        }
        if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW) {
            if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum))
                continue;
        }
        else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FOREGROUND_SALIENCY) {
            if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, false))
                continue;
        }
        else {
            if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true))
                continue;
        }

        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset1 = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(prefetch_data + offset1);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
        trans_time += timer.MicroSeconds();

        prefetch_label[item_id] = lines_[lines_id_].second;
        //LOG()

        //next iteration
        lines_id_++;
        if (lines_id_ >= lines_size) {
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if(this->layer_param_.video_data_param().shuffle()){
                ShuffleVideos();
            }
        }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}       // namespace caffe
#endif  // USE_OPENCV
