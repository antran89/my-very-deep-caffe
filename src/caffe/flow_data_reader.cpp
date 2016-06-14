#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/flow_data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<FlowDataReader::Body> > FlowDataReader::bodies_;
static boost::mutex bodies_mutex_;

FlowDataReader::FlowDataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.flow_data_param().prefetch() * param.flow_data_param().batch_size())) {
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

FlowDataReader::~FlowDataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

FlowDataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

FlowDataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

FlowDataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {

    // initialize some private variables
    new_length_ = param.flow_data_param().new_length();
    num_segments_ = param.flow_data_param().num_segments();
    switch (param.flow_data_param().modality()) {
    case FlowDataParameter_Modality_FLOW:
        fr_channels_ = 2;
        new_channels_ = fr_channels_ * new_length_ * num_segments_;
        break;
    case FlowDataParameter_Modality_RGB:
        fr_channels_ = 3;
        new_channels_ = fr_channels_ * new_length_ * num_segments_;
        break;
    default:
        break;
    }

    phase_ = param.phase();
    if (phase_ == TRAIN) {
        const unsigned int rng_seed = caffe_rng_rand();
        rng_.reset(new Caffe::RNG(rng_seed));
    } else
        rng_.reset();

    // start threads
    StartInternalThread();
}

FlowDataReader::Body::~Body() {
  StopInternalThread();
}

void FlowDataReader::Body::InternalThreadEntry() {
  FlowDataParameter_DB backend = param_.flow_data_param().backend();
  string backend_str;
  switch (backend)
  {
  case FlowDataParameter_DB_LEVELDB:
      backend_str = "leveldb";
      break;
  case FlowDataParameter_DB_LMDB:
      backend_str = "lmdb";
      break;
  }
  shared_ptr<db::DB> db(db::GetDB(backend_str));
  db->Open(param_.flow_data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
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

void FlowDataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

// modify read_one function to trim video into a length
void FlowDataReader::Body::read_one_varied_length_datum(db::Cursor* cursor, QueuePair* qp) {
  CPUTimer read_one_timer;
  read_one_timer.Start();
  CPUTimer timer;
  timer.Start();

  Datum* datum = new Datum(); // storing a raw video datum
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  double deserialize_time = timer.MicroSeconds();

  // temporally trim videos into new_length videos
  Datum* trimmed_datum = qp->free_.pop("Waiting for free datum");
  trimmed_datum->set_channels(new_channels_);
  trimmed_datum->set_height(datum->height());
  trimmed_datum->set_width(datum->width());
  trimmed_datum->set_label(datum->label());
  int channel_size = fr_channels_ * new_length_ * datum->height() * datum->width();

  // generate offset values
  int video_length = datum->channels()/fr_channels_;
  int average_duration = video_length / num_segments_;
  vector<int> offsets;
  for (int i = 0; i < num_segments_; i++) {
      if (phase_ == TRAIN) {
          caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(rng_->generator());
          int offset = (*frame_rng)() % (average_duration - new_length_ + 1);
          offsets.push_back(offset + i * average_duration);
      } else
          offsets.push_back(int((average_duration-new_length_+1)/2 + i * average_duration));
  }

  timer.Start();
  // copy data to new datum
  string *datum_string = datum->mutable_data();
  string buffer;
  for (int i = 0; i < num_segments_; i++) {
      int offset = offsets[i];
      int mem_offset = offset * fr_channels_ * datum->height() * datum->width();
      buffer.append(*datum_string, mem_offset, channel_size);
  }
  trimmed_datum->set_data(buffer);
  qp->full_.push(trimmed_datum);

  double copy_datum_time = timer.MicroSeconds();

  timer.Start();
  // release temporary datum
  delete datum;
  double delete_datum_time = timer.MicroSeconds();

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }

  read_one_timer.Stop();
  DLOG(INFO) << "Read one datum time: " << read_one_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "   Deserialize time: " << deserialize_time/1000 << " ms.";
  DLOG(INFO) << "       Copying time: " << copy_datum_time/1000 << "ms.";
  DLOG(INFO) << " Release datum time: " << delete_datum_time/1000 << "ms.";

}

}  // namespace caffe
