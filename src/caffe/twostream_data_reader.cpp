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
#include "caffe/twostream_data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<TwostreamDataReader::Body> > TwostreamDataReader::bodies_;
static boost::mutex bodies_mutex_;

TwostreamDataReader::TwostreamDataReader(const LayerParameter& param)
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

TwostreamDataReader::~TwostreamDataReader() {
    string key = source_key(body_->param_);
    body_.reset();
    boost::mutex::scoped_lock lock(bodies_mutex_);
    if (bodies_[key].expired()) {
        bodies_.erase(key);
    }
}

//

TwostreamDataReader::QueuePair::QueuePair(int size) {
    // Initialize the free queue with requested number of datums
    for (int i = 0; i < size; ++i) {
        rgb_free_.push(new Datum());
        flow_free_.push(new Datum());
    }
}

TwostreamDataReader::QueuePair::~QueuePair() {
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

TwostreamDataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
    StartInternalThread();
}

TwostreamDataReader::Body::~Body() {
    StopInternalThread();
}

void TwostreamDataReader::Body::InternalThreadEntry() {
    TwostreamDataParameter_DB backend = param_.twostream_data_param().backend();
    string backend_str;
    switch (backend)
    {
    case TwostreamDataParameter_DB_LEVELDB:
        backend_str = "leveldb";
        break;
    case TwostreamDataParameter_DB_LMDB:
        backend_str = "lmdb";
        break;
    }

    shared_ptr<db::DB> flow_db(db::GetDB(backend_str));
    flow_db->Open(param_.twostream_data_param().flow_source(), db::READ);
    shared_ptr<db::Cursor> flow_cursor(flow_db->NewCursor());
    shared_ptr<db::DB> rgb_db(db::GetDB(backend_str));
    rgb_db->Open(param_.twostream_data_param().rgb_source(), db::READ);
    shared_ptr<db::Cursor> rgb_cursor(rgb_db->NewCursor());

    vector<shared_ptr<QueuePair> > qps;
    try {
        int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

        // To ensure deterministic runs, only start running once all solvers
        // are ready. But solvers need to peek on one item during initialization,
        // so read one item, then wait for the next solver.
        for (int i = 0; i < solver_count; ++i) {
            shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
            read_one(flow_cursor.get(), rgb_cursor.get(), qp.get());
            qps.push_back(qp);
        }
        // Main loop
        while (!must_stop()) {
            for (int i = 0; i < solver_count; ++i) {
                read_one(flow_cursor.get(), rgb_cursor.get(), qps[i].get());
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

void TwostreamDataReader::Body::read_one(db::Cursor* flow_cursor, db::Cursor* rgb_cursor, QueuePair* qp) {
    // reading flow datum
    Datum* flow_datum = qp->flow_free_.pop();
    // TODO deserialize in-place instead of copy?
    flow_datum->ParseFromString(flow_cursor->value());
    qp->flow_full_.push(flow_datum);
    // reading rgb datum
    Datum* rgb_datum = qp->rgb_free_.pop();
    rgb_datum->ParseFromString(rgb_cursor->value());
    qp->rgb_full_.push(rgb_datum);

    // go to the next iter
    flow_cursor->Next();
    rgb_cursor->Next();
    if (!flow_cursor->valid()) {
        CHECK_EQ(rgb_cursor->valid(), false) << "RGB and flow database must be in the same length";
        DLOG(INFO) << "Restarting data prefetching from start.";
        flow_cursor->SeekToFirst();
        rgb_cursor->SeekToFirst();
    }
}

}  // namespace caffe
