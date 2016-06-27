#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
//#ifdef USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//#endif

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean image of a set of color flow images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_color_flow_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE] [MEAN_IMAGE_FILENAME]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_color_flow_image_mean");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  BlobProto sum_blob;
  int count = 0;
  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());

  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    DecodeDatumNative(&datum);

    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc >= 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  // compute the mean image in a sum_blob.
  const int num_channels = sum_blob.channels();
  const int height = sum_blob.height();
  const int width = sum_blob.width();
  const int dim = sum_blob.height() * sum_blob.width();
  assert(num_channels % 3 == 0);
  const int num_frames = sum_blob.channels()/3;
  LOG(INFO) << "Number of frames in the mean blob: " << num_frames;

  cv::Mat mean_image = cv::Mat::zeros(sum_blob.height(), sum_blob.width(), CV_32FC3);
  for (int c = 0; c < num_channels; c++) {
      int ch = c % 3;
      int start_ind = dim * c;
      int ind = 0;
      // add each pixels into the corresponding channel
      // Why C++ does not have unified matrix notations such as in Torch Tensor, Matlab Mat
      for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
              mean_image.at<cv::Vec3f>(h, w)[ch] += sum_blob.data(start_ind + ind);
              ind++;
          }
      }
  }
  // compute the mean values in the mean image
  // an efficient way to access OpenCV Mat
  // The other way is mean_image.at<cv::Vec3f>(h, w)[ch].
  // This is not so important, just an programming exercise
  // Please use it in the critical part of your code.
  for (int h = 0; h < height; h++) {
      float *ptr = mean_image.ptr<float>(h);
      for (int w = 0; w < width*3; w++) {
          ptr[w] = ptr[w]/num_frames;
      }
  }

  // saving the image
  if (argc == 4) {
      LOG(INFO) << "Write mean image to " << argv[3];
      cv::Mat save_img;
      mean_image.convertTo(save_img, CV_8UC3);
      cv::imwrite(argv[3], save_img);
  }

  std::vector<float> mean_values(3, 0.0);
  for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
              mean_values[c] += mean_image.at<cv::Vec3f>(h, w)[c];
          }
      }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
