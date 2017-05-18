/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/twostream_snippet_data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<TwostreamSnippetDataReader::Body> > TwostreamSnippetDataReader::bodies_;
static boost::mutex bodies_mutex_;

TwostreamSnippetDataReader::TwostreamSnippetDataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
                                  param.twostream_data_param().prefetch() * param.twostream_data_param().batch_size())) {
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

TwostreamSnippetDataReader::~TwostreamSnippetDataReader() {
    string key = source_key(body_->param_);
    body_.reset();
    boost::mutex::scoped_lock lock(bodies_mutex_);
    if (bodies_[key].expired()) {
        bodies_.erase(key);
    }
}

//

TwostreamSnippetDataReader::QueuePair::QueuePair(int size) {
    // Initialize the free queue with requested number of datums
    for (int i = 0; i < size; ++i) {
        rgb_free_.push(new Datum());
        flow_free_.push(new Datum());
    }
}

TwostreamSnippetDataReader::QueuePair::~QueuePair() {
    Datum* datum;
    while (rgb_free_.try_pop(&datum)) {
        delete datum;
    }
    while (rgb_full_.try_pop(&datum)) {
        delete datum;
    }
    while (flow_free_.try_pop(&datum)) {
        delete datum;
    }
    while (flow_full_.try_pop(&datum)) {
        delete datum;
    }
}

//

TwostreamSnippetDataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
    StartInternalThread();
}

TwostreamSnippetDataReader::Body::~Body() {
    StopInternalThread();
}

void TwostreamSnippetDataReader::Body::InternalThreadEntry() {
    const string flow_source = param_.twostream_data_param().flow_source();
    const string rgb_source = param_.twostream_data_param().rgb_source();
    bool preserve_temporal = param_.twostream_data_param().preserve_temporal();
    int new_length = param_.twostream_data_param().new_length();
    std::ifstream inflow_file(flow_source.c_str());
    std::ifstream inrgb_file(rgb_source.c_str());

    vector<shared_ptr<QueuePair> > qps;
    try {
        int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

        // To ensure deterministic runs, only start running once all solvers
        // are ready. But solvers need to peek on one item during initialization,
        // so read one item, then wait for the next solver.
        for (int i = 0; i < solver_count; ++i) {
            shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
            read_one(inflow_file, inrgb_file, preserve_temporal, new_length, qp.get());
            qps.push_back(qp);
        }
        // Main loop
        while (!must_stop()) {
            for (int i = 0; i < solver_count; ++i) {
                read_one(inflow_file, inrgb_file, preserve_temporal, new_length, qps[i].get());
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

void TwostreamSnippetDataReader::Body::read_one(std::ifstream& inflow_file, std::ifstream& inrgb_file, const bool preserve_temporal,
                                                const int new_length, QueuePair* qp) {
    // reading flow and rgb datum
    Datum* flow_datum = qp->flow_free_.pop();
    Datum* rgb_datum = qp->rgb_free_.pop();
    // reading a video snippet into flow and rgb datum
    string file_name;
    int start_fr, label;
    if (inflow_file >> file_name >> start_fr >> label) {
        bool status;
        vector<int> offsets(1, 0);    // assuming only 1 segment in each video.
        offsets[0] = start_fr - 1;
        if (preserve_temporal)
            status = ReadSegmentFlowToTemporalDatum(file_name, label, offsets, 0, 0, new_length, flow_datum);
        else
            status = ReadSegmentFlowToDatum(file_name, label, offsets, 0, 0, new_length, flow_datum);
        if (status == false)
            LOG(FATAL) << "Failed to read flows from file: " <<  file_name;
        // two flow and rgb txt file are corresponding, no need to check second time
        inrgb_file >> file_name >> start_fr >> label;
        if (preserve_temporal)
            status = ReadSegmentRGBToTemporalDatum(file_name, label, offsets, 0, 0, new_length, rgb_datum);
        else
            status = ReadSegmentRGBToDatum(file_name, label, offsets, 0, 0, new_length, rgb_datum);

        if (status == false)
            LOG(FATAL) << "Failed to read rgb frames from file: " <<  file_name;

        qp->flow_full_.push(flow_datum);
        qp->rgb_full_.push(rgb_datum);
    }
    else {
        qp->flow_free_.push(flow_datum);
        qp->rgb_free_.push(rgb_datum);
        LOG(INFO) << "Failed to read one datum.";
    }

    // check if end of file, then rewind
    if (inflow_file.peek() == EOF) {
        inrgb_file >> file_name >> start_fr >> label;   // move inrgb_file peek into the end
        CHECK_EQ(inrgb_file.peek(), EOF) << "RGB and flow database must be in the same length";
        DLOG(INFO) << "Restarting data prefetching from start.";
        inflow_file.clear();
        inflow_file.seekg(0, ios::beg);
        inrgb_file.clear();
        inrgb_file.seekg(0, ios::beg);
    }
}

}  // namespace caffe
