#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
    return feature_extraction_pipeline<float>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
    string net_proto = string(argv[1]);
    string pretrained_model = string(argv[2]);
    int device_id = atoi(argv[3]);
    uint batch_size = atoi(argv[4]);
    uint num_mini_batches = atoi(argv[5]);
    char* fn_feat = argv[6];

    //  Caffe::set_phase(Caffe::TEST);
    if (device_id>=0){
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
        LOG(INFO) << "Using GPU #" << device_id;
    }
    else{
        Caffe::set_mode(Caffe::CPU);
        LOG(INFO) << "Using CPU";
    }

    shared_ptr<Net<Dtype> > feature_extraction_net(
                new Net<Dtype>(net_proto, caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_model);

    for (int i=7; i<argc; i++){
        CHECK(feature_extraction_net->has_blob(string(argv[i])))
                << "Unknown feature blob name " << string(argv[i])
                << " in the network " << string(net_proto);
    }

    LOG(INFO) << "Extracting features for " << num_mini_batches << " batches.";
    std::ifstream infile(fn_feat);
    string feat_prefix;
    std::vector<string> list_prefix;
    std::vector<string> lines;
    while (infile >> feat_prefix)
        lines.push_back(feat_prefix);
    infile.close();
    const int num_lines = lines.size();

    LOG(INFO) << "Extracting " << num_lines << " features.";

    int image_index = 0;
    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
        feature_extraction_net->Forward();

        list_prefix.clear();
        for (int n = 0; n < batch_size; n++){
            int line_index = image_index + n;
            if (line_index < num_lines)
                list_prefix.push_back(lines[line_index]);
            else
                break;
        }
        if (list_prefix.empty())
            break;

        for (int k=7; k<argc; k++) {
            const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
                    ->blob_by_name(string(argv[k]));
            int num_features = feature_blob->num();
            CHECK_EQ(num_features, batch_size) << "Number of features in a batch must be equal to batch size";

            for (int n = 0; n < num_features; ++n) {
                if (list_prefix.size() > n) {
                    // This is a bad condition! It might be wrong if the batch_size argument in cmd line is set wrong.
                    // But this condition is good for guarding again the last features that are not enough for a batch.
                    // The feature extraction of last batch is correct only when using with LDMB database, because of
                    // load_batch() function.
                    string fn_feat = list_prefix[n] + string(".") + string(argv[k]);
                    save_blob_to_binary(feature_blob.get(), fn_feat, n);
                }
            }
        }
        image_index += list_prefix.size();
        if (batch_index % 100 == 0) {
            LOG(INFO)<< "Extracted features of " << image_index <<
                        " images.";
        }
    }
    LOG(INFO)<< "Successfully extracted " << image_index << " features!";
    return 0;
}
