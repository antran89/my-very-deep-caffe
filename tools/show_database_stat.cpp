/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

// This program show some statistics of a lmdb/leveldb
// Usage:
//   show_database_stat [FLAGS] DB_NAME
//
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/db_lmdb.hpp"
#include <lmdb.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

const size_t LMDB_MAP_SIZE = 1099511627776;  // 1 TB

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of video chunk to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    show_database_stat [FLAGS] DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/show_database_stat");
    return 1;
  }

//  scoped_ptr<db::DB> db(db::getDB(FLAGS_backend));
//  db->Open(argv[1], db::READ);

  MDB_txn* mdb_txn;
  MDB_env* mdb_env;
  MDB_dbi mdb_dbi;

  int flags = MDB_RDONLY | MDB_NOTLS;

  // open the database handle
  db::MDB_CHECK(mdb_env_create(&mdb_env));
  db::MDB_CHECK(mdb_env_set_mapsize(mdb_env, LMDB_MAP_SIZE));
  db::MDB_CHECK(mdb_env_open(mdb_env, argv[1], flags, 0664));

  // create a transaction handle
  db::MDB_CHECK(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn));
  db::MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

  // retrieve statistics for a database
  MDB_stat stat;
  db::MDB_CHECK(mdb_stat(mdb_txn, mdb_dbi, &stat));

  // print out the statistics
  LOG(INFO) << "Some statistics of the database: " << argv[1];
  LOG(INFO) << "Size of a database page: " << stat.ms_psize;
  LOG(INFO) << "Depth (height) of the B-tree: " << stat.ms_depth;
  LOG(INFO) << "Number of leaf pages: " << stat.ms_leaf_pages;
  LOG(INFO) << "Number of overflow pages: " << stat.ms_overflow_pages;
  LOG(INFO) << "Number of data items: " << stat.ms_entries;

  return 0;
}


