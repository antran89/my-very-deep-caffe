#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/video_snippet_data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<VideoSnippetDataReader::Body> > VideoSnippetDataReader::bodies_;
static boost::mutex bodies_mutex_;

VideoSnippetDataReader::VideoSnippetDataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.video_snippet_data_param().prefetch() * param.video_snippet_data_param().batch_size())) {
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

VideoSnippetDataReader::~VideoSnippetDataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

VideoSnippetDataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

VideoSnippetDataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

VideoSnippetDataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

VideoSnippetDataReader::Body::~Body() {
  StopInternalThread();
}

void VideoSnippetDataReader::Body::InternalThreadEntry() {
  const string source = param_.video_snippet_data_param().source();
  bool preserve_temporal = param_.video_snippet_data_param().preserve_temporal();
  int new_length = param_.video_snippet_data_param().new_length();
  bool is_flow = param_.video_snippet_data_param().modality() == VideoSnippetDataParameter_Modality_FLOW;
  std::ifstream infile(source.c_str());
  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(infile, preserve_temporal, is_flow, new_length, qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(infile, preserve_temporal, is_flow, new_length, qps[i].get());
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

void VideoSnippetDataReader::Body::read_one(std::ifstream& infile, const bool preserve_temporal, const bool is_flow,
                                            const bool new_length, QueuePair* qp) {
  Datum* datum = qp->free_.pop();
  // reading a video snippet into datum
  string file_name;
  int start_fr, label;
  if (infile >> file_name >> start_fr >> label) {
      bool status;
      vector<int> offsets(1, 0);    // assuming only 1 segment in each video.
      offsets[0] = start_fr - 1;
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
      LOG(INFO) << "Failed to read one datum.";
  }

  // check if end of file, then rewind
  if (infile.peek() == EOF) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      infile.clear();
      infile.seekg(0, ios::beg);
  }
}

}  // namespace caffe
