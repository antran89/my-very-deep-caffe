/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */

#include <stdio.h>  // for snprintf
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
#include "caffe/layers/video_test_data_layer.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template <typename Dtype>
bool save_average_features_to_binary(Blob<Dtype>* blob, const string fn_blob, int num_index);

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
    int num_expected_features = caffe::CAFFE_NUM_TEST_VIEWS * batch_size;

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

    LOG(INFO)<< "Extracting features for " << num_mini_batches << " batches";
    std::ifstream infile(fn_feat);
    string feat_prefix;
    std::vector<string> list_prefix;

    int image_index = 0;

    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
        feature_extraction_net->Forward();
        list_prefix.clear();
        for (int n=0; n<batch_size; n++){
            if (infile >> feat_prefix)
                list_prefix.push_back(feat_prefix);
            else
                break;
        }

        if (list_prefix.empty())
            break;
        for (int k=7; k<argc; k++){
            const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
                    ->blob_by_name(string(argv[k]));
            int num_features = feature_blob->shape(0);
            CHECK_EQ(num_features, num_expected_features);

            // save average features into a binary file
            for (int n = 0; n < batch_size; ++n) {
                if (list_prefix.size()>n) {
                    // This is a bad condition! It might be wrong if the batch_size argument in cmd line is set wrong.
                    // But this condition is good for guarding again the last features that are not enough for a batch.
                    // The feature extraction of last batch is correct only when using with LDMB database, because of
                    // load_batch() function.
                    string fn_feat = list_prefix[n] + string(".") + string(argv[k]);
                    //save_blob_to_binary(feature_blob.get(), fn_feat, n);
                    save_average_features_to_binary(feature_blob.get(), fn_feat, n);
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
    infile.close();
    return 0;
}


template <>
bool save_average_features_to_binary<float>(Blob<float>* blob, const string fn_blob, int num_index)
{
    FILE *f;
    float *buff;
    int num, channel, length, width, height;
    f = fopen(fn_blob.c_str(), "wb");
    if (f==NULL)
        return false;

    const int num_views = caffe::CAFFE_NUM_TEST_VIEWS;

    if (num_index<0){
        num = blob->shape(0) / num_views;   // [IMPORTANT] divide num_views number
        buff = blob->mutable_cpu_data();
    } else {
        num = 1;
        vector<int> indices(1, 0);
        // [IMPORTANT] need to multiply num_views to have correct indexes
        indices[0] = num_index * num_views;
        buff = blob->mutable_cpu_data() + blob->offset(indices);
    }
    channel = blob->shape(1);
    if (blob->shape().size() > 2) {
        length = blob->shape(2);
        height = blob->shape(3);
        width = blob->shape(4);
    } else {
        length = 1;
        height = 1;
        width = 1;
    }

    int avg_buff_size = num * channel * length * height * width;

    // average features from different views
    float *avg_buff = new float[avg_buff_size];
    for (int n = 0; n < num; n++)
        for (int c = 0; c < channel; c++)
            for (int l = 0; l < length; l++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        int avg_index = ((n * channel + c) * height + h) * width + w;
                        float sum_val = 0;
                        for (int v = 0; v < num_views; v++)
                        {
                            int buff_index = (((n * num_views + v) * channel + c) * height + h) * width + w;
                            sum_val += buff[buff_index];
                        }
                        // avaraging
                        avg_buff[avg_index] = sum_val / num_views;
                    }

    fwrite(&num, sizeof(int), 1, f);
    fwrite(&channel, sizeof(int), 1, f);
    fwrite(&length, sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    fwrite(&width, sizeof(int), 1, f);
    fwrite(avg_buff, sizeof(float), avg_buff_size, f);
    fclose(f);
    delete avg_buff;

    return true;
}

