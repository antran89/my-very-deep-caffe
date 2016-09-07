// This program is to show first image from a LMDB database.
// "Usage:\n"
// "      visualize_database_image [FLAGS] INPUT_DB\n");

#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

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
DEFINE_bool(gray, true,
            "When this option is on, treat images as grayscale ones");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Compute the mean image of a set of flow images given by"
                            " a leveldb/lmdb\n"
                            "Usage:\n"
                            "    show_database_image [FLAGS] INPUT_DB \n"
                            "    Press ESC or q or Q key to quit.");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 2 || argc > 4) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/show_database_image");
        return 1;
    }

    const bool is_color = !FLAGS_gray;
    if (is_color)
        LOG(INFO) << "Show images from color image database.";
    else
        LOG(INFO) << "Show images from gray image database.";

    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open(argv[1], db::READ);
    scoped_ptr<db::Cursor> cursor(db->NewCursor());

    // load first datum
    Datum datum;
    datum.ParseFromString(cursor->value());

    if (DecodeDatumNative(&datum)) {
        LOG(INFO) << "Decoding Datum";
    }

    const int height = datum.height();
    const int width = datum.width();
    const int data_size = datum.channels() * datum.height() * datum.width();
    int size_in_datum;

    LOG(INFO) << "Height and width of image in database: (" << height << ", " << width << ")";

    const int image_type = is_color ? CV_32FC3 : CV_32FC1;
    cv::Mat image = cv::Mat::zeros(height, width, image_type);
    const int image_channels = is_color ? 3 : 1;

    LOG(INFO) << "Starting iteration";
    while (cursor->valid()) {
        Datum datum;
        datum.ParseFromString(cursor->value());
        DecodeDatumNative(&datum);

        const std::string& data = datum.data(); // need to cast to access data elements
        size_in_datum = std::max<int>(datum.data().size(),
                                      datum.float_data_size());
        CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
                                              size_in_datum;
        // fill content of the first image
        int ind = 0;
        for (int c = 0; c < image_channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    if (data.size() != 0) {
                        CHECK_EQ(data.size(), size_in_datum);
                        if (is_color)
                            image.at<cv::Vec3f>(h, w)[c] = static_cast<uint8_t>(data[ind]);
                        else
                            image.at<float>(h, w) = static_cast<uint8_t>(data[ind]);
                    }
                    else {
                        CHECK_EQ(datum.float_data_size(), size_in_datum);
                        if (is_color)
                            image.at<cv::Vec3f>(h, w)[c] = static_cast<float>(datum.float_data(ind));
                        else
                            image.at<float>(h, w) = static_cast<float>(datum.float_data(ind));
                    }
                    ind++;
                }
            }
        }

        // show the image
        cv::Mat vis_image;
        int vis_image_type = is_color ? CV_8UC3 : CV_8UC1;
        image.convertTo(vis_image, vis_image_type);
        cv::imshow("The first image in a datum", vis_image);
        char key = (char)cv::waitKey();
        if (key == 27 || key == 'q' || key == 'Q')
            break;

        cursor->Next();
    }

#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
    return 0;
}
