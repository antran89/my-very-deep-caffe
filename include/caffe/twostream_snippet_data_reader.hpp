/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef CAFFE_TWOSTREAM_SNIPPET_DATA_READER_HPP_
#define CAFFE_TWOSTREAM_SNIPPET_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads two-stream (rgb & flow) data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
class TwostreamSnippetDataReader {
public:
    explicit TwostreamSnippetDataReader(const LayerParameter& param);
    ~TwostreamSnippetDataReader();

    inline BlockingQueue<Datum*>& rgb_free() const {
        return queue_pair_->rgb_free_;
    }
    inline BlockingQueue<Datum*>& rgb_full() const {
        return queue_pair_->rgb_full_;
    }
    inline BlockingQueue<Datum*>& flow_free() const {
        return queue_pair_->flow_free_;
    }
    inline BlockingQueue<Datum*>& flow_full() const {
        return queue_pair_->flow_full_;
    }

protected:
    // Queue pairs are shared between a body and its readers
    class QueuePair {
    public:
        explicit QueuePair(int size);
        ~QueuePair();

        BlockingQueue<Datum*> rgb_free_;
        BlockingQueue<Datum*> rgb_full_;
        BlockingQueue<Datum*> flow_free_;
        BlockingQueue<Datum*> flow_full_;

        DISABLE_COPY_AND_ASSIGN(QueuePair);
    };

    // A single body is created per source
    class Body : public InternalThread {
    public:
        explicit Body(const LayerParameter& param);
        virtual ~Body();

    protected:
        void InternalThreadEntry();
        void read_one(std::ifstream& inflow_file, std::ifstream& inrgb_file, const bool preserve_temporal,
                      const bool new_length, QueuePair* qp);

        const LayerParameter param_;
        BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

        friend class TwostreamSnippetDataReader;

        DISABLE_COPY_AND_ASSIGN(Body);
    };

    // A source is uniquely identified by its layer name + path, in case
    // the same database is read from two different locations in the net.
    static inline string source_key(const LayerParameter& param) {
        return param.name() + ":" + param.twostream_data_param().flow_source();
    }

    const shared_ptr<QueuePair> queue_pair_;
    shared_ptr<Body> body_;

    static map<const string, boost::weak_ptr<TwostreamSnippetDataReader::Body> > bodies_;

    DISABLE_COPY_AND_ASSIGN(TwostreamSnippetDataReader);
};

}  // namespace caffe

#endif  // CAFFE_TWOSTREAM_SNIPPET_DATA_READER_HPP_
