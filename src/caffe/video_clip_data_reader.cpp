#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/video_clip_data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<VideoClipDataReader::Body> > VideoClipDataReader::bodies_;
static boost::mutex bodies_mutex_;

VideoClipDataReader::VideoClipDataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
                                  param.video_data_param().prefetch() * param.video_data_param().batch_size())) {
    // Get or create a body
    boost::mutex::scoped_lock lock(bodies_mutex_);
    string key = source_key(param);
    weak_ptr<Body>& weak = bodies_[key];
    body_ = weak.lock();
    if (!body_) {
        body_.reset(new Body(param));
        bodies_[key] = weak_ptr<Body>(body_);
    }
    body_->new_queue_pairs_.push(queue_pair_);
}

VideoClipDataReader::~VideoClipDataReader() {
    string key = source_key(body_->param_);
    body_.reset();
    boost::mutex::scoped_lock lock(bodies_mutex_);
    if (bodies_[key].expired()) {
        bodies_.erase(key);
    }
}

//

VideoClipDataReader::QueuePair::QueuePair(int size) {
    // Initialize the free queue with requested number of datums
    for (int i = 0; i < size; ++i) {
        free_.push(new Datum());
    }
}

VideoClipDataReader::QueuePair::~QueuePair() {
    Datum* datum;
    while (free_.try_pop(&datum)) {
        delete datum;
    }
    while (full_.try_pop(&datum)) {
        delete datum;
    }
}

//

VideoClipDataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {

    // initialize random number generator
    const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
    frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

    StartInternalThread();
}

VideoClipDataReader::Body::~Body() {
    StopInternalThread();
}

void VideoClipDataReader::Body::InternalThreadEntry() {
    const string source = param_.video_data_param().source();
    bool preserve_temporal = param_.video_data_param().preserve_temporal();
    int new_length = param_.video_data_param().new_length();
    bool is_flow = param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW;
    int num_segments = param_.video_data_param().num_segments();
    std::ifstream infile;
    infile.open(source.c_str());
    if (!infile.is_open())
        LOG(FATAL) << "Failed to open the file: " << source;

    vector<shared_ptr<QueuePair> > qps;
    try {
        int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

        // To ensure deterministic runs, only start running once all solvers
        // are ready. But solvers need to peek on one item during initialization,
        // so read one item, then wait for the next solver.
        for (int i = 0; i < solver_count; ++i) {
            shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
            read_one(infile, preserve_temporal, is_flow, new_length, num_segments, qp.get());
            qps.push_back(qp);
        }
        // Main loop
        while (!must_stop()) {
            for (int i = 0; i < solver_count; ++i) {
                read_one(infile, preserve_temporal, is_flow, new_length, num_segments, qps[i].get());
            }
            // Check no additional readers have been created. This can happen if
            // more than one net is trained at a time per process, whether single
            // or multi solver. It might also happen if two data layers have same
            // name and same source.
            CHECK_EQ(new_queue_pairs_.size(), 0);
        }
    } catch (boost::thread_interrupted&) {
        // Interrupted exception is expected on shutdown
    }
}

void VideoClipDataReader::Body::read_one(std::ifstream& infile, const bool preserve_temporal, const bool is_flow,
                                         const int new_length, const int num_segments, QueuePair* qp) {
    Datum* datum = qp->free_.pop();
    // reading a video snippet into datum
    string file_name;
    int vid_length, label;
    if (infile >> file_name >> vid_length >> label) {
        bool status;
        vector<int> offsets;
        int average_duration = vid_length / num_segments;
        for (int i = 0; i < num_segments; ++i) {
            if (this->param_.phase() == TRAIN) {
                if (average_duration >= new_length) {
                    caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
                    int offset = (*frame_rng)() % (average_duration - new_length + 1);
                    offsets.push_back(offset + i*average_duration);
                } else {
                    offsets.push_back(0);
                }
            } else {
                if (average_duration >= new_length)
                    offsets.push_back(int((average_duration - new_length + 1)/2 + i * average_duration));
                else
                    offsets.push_back(0);
            }
        }

        if (is_flow) {
            if (preserve_temporal)
                status = ReadSegmentFlowToTemporalDatum(file_name, label, offsets, 0, 0, new_length, datum);
            else
                status = ReadSegmentFlowToDatum(file_name, label, offsets, 0, 0, new_length, datum);
        } else {
            if (preserve_temporal)
                status = ReadSegmentRGBToTemporalDatum(file_name, label, offsets, 0, 0, new_length, datum);
            else
                status = ReadSegmentRGBToDatum(file_name, label, offsets, 0, 0, new_length, datum);
        }

        if (status == false)
            LOG(FATAL) << "Failed to read data from file: " <<  file_name;
        qp->full_.push(datum);
    }
    else {
        qp->free_.push(datum);
        DLOG(INFO) << "Failed to read one datum.";
    }

    // check if end of file, then rewind
    if (infile.peek() == EOF) {
        LOG(INFO) << "Restarting data prefetching from start.";
        infile.clear();
        infile.seekg(0, ios::beg);
    }
}

}   // namespace caffe
