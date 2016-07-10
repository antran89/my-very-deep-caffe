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

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


namespace caffe {


void ImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
	for (int c = 0; c < 3; ++c) {
	  for (int h = 0; h < img->rows; ++h) {
		for (int w = 0; w < img->cols; ++w) {
			buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	  }
	}
}
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c){
    int idx = 0;
	for (int h = 0; h < img->rows; ++h) {
	    for (int w = 0; w < img->cols; ++w) {
		    buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	}
}

void GrayImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
    for (int h = 0; h < img->rows; ++h) {
	  for (int w = 0; w < img->cols; ++w) {
		buffer[idx++] = img->at<unsigned char>(h, w);
	  }
	}
}
void BufferToGrayImage(const char* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = buffer[idx++];
	  }
	}
}
void BufferToColorImage(const char* buffer, const int height, const int width, cv::Mat* img){
	img->create(height, width, CV_8UC3);
	for (int c=0; c<3; c++) {
		for (int h = 0; h < height; ++h) {
		  for (int w = 0; w < width; ++w) {
			img->at<cv::Vec3b>(h, w)[c] = buffer[c * width * height + h * width + w];
		  }
		}
	}
}

template <>
bool load_blob_from_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

    vector<int> shape(5);
    shape[0] = n;
    shape[1] = c;
    shape[2] = l;
    shape[3] = h;
    shape[4] = w;
    blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

    vector<int> shape(5);
    shape[0] = n;
    shape[1] = c;
    shape[2] = l;
    shape[3] = h;
    shape[4] = w;
    blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_uint8_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

    vector<int> shape(5);
    shape[0] = n;
    shape[1] = c;
    shape[2] = l;
    shape[3] = h;
    shape[4] = w;
    blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (float)temp_buff[i];

	delete []temp_buff;
	return true;
}

template <>
bool load_blob_from_uint8_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

    vector<int> shape(5);
    shape[0] = n;
    shape[1] = c;
    shape[2] = l;
    shape[3] = h;
    shape[4] = w;
    blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (double)temp_buff[i];

	delete []temp_buff;
	return true;
}


template <>
bool save_blob_to_binary<float>(Blob<float>* blob, const string fn_blob, int num_index) {
	FILE *f;
	float *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

    if (num_index<0){
        n = blob->shape(0);
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
        vector<int> indices(1, 0);
        indices[0] = num_index;
        buff = blob->mutable_cpu_data() + blob->offset(indices);
	}
    c = blob->shape(1);
    if (blob->shape().size() > 2) {
        l = blob->shape(2);
        h = blob->shape(3);
        w = blob->shape(4);
    } else {
        l = 1;
        h = 1;
        w = 1;
    }

	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool save_blob_to_binary<double>(Blob<double>* blob, const string fn_blob, int num_index) {
	FILE *f;
	double *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	if (num_index<0){
        n = blob->shape(0);
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
        vector<int> indices(1, 0);
        indices[0] = num_index;
        buff = blob->mutable_cpu_data() + blob->offset(indices);
	}
    c = blob->shape(1);
    if (blob->shape().size() > 2) {
        l = blob->shape(2);
        h = blob->shape(3);
        w = blob->shape(4);
    } else {
        l = 1;
        h = 1;
        w = 1;
    }

	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}
}
