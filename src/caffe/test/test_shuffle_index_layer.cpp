#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/shuffle_index_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class ShuffleIndexLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ShuffleIndexLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ShuffleIndexLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;
    ShuffleIndexParameter* shuffle_index_param = layer_param.mutable_shuffle_index_param();

    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      blob_bottom_->mutable_cpu_data()[i] = i;
    }

    int shuffle_index[] = { 1, 3, 0, 2 };
    for (int i = 0; i < 4; i++)
        shuffle_index_param->add_new_index(shuffle_index[i]);

    ShuffleIndexLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), blob_bottom_->shape(shuffle_index[0]));
    EXPECT_EQ(blob_top_->shape(1), blob_bottom_->shape(shuffle_index[1]));
    EXPECT_EQ(blob_top_->shape(2), blob_bottom_->shape(shuffle_index[2]));
    EXPECT_EQ(blob_top_->shape(3), blob_bottom_->shape(shuffle_index[3]));

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    int bot_channels = blob_bottom_->shape(1);
    int bot_height = blob_bottom_->shape(2);
    int bot_width = blob_bottom_->shape(3);
    vector<int> bot_index(4), top_index(4);
    for (int index = 0; index < blob_bottom_->count(); ++index) {
      bot_index[0] = index / (bot_channels * bot_width * bot_height);
      bot_index[1] = (index / (bot_width * bot_height)) % bot_channels;
      bot_index[2] = (index / bot_width) % bot_height;
      bot_index[3] = index % bot_width;
      for (int k = 0; k < 4; k++)
          top_index[k] = bot_index[shuffle_index[k]];
      EXPECT_EQ(
          blob_bottom_->cpu_data()[index],
          blob_top_->cpu_data()[blob_top_->offset(top_index)]);
    }
  }
};

TYPED_TEST_CASE(ShuffleIndexLayerTest, TestDtypesAndDevices);

TYPED_TEST(ShuffleIndexLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(ShuffleIndexLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  // add index into layer parameter & create ShuffleIndexLayer
  ShuffleIndexParameter* shuffle_index_param = layer_param.mutable_shuffle_index_param();
  int shuffle_index[] = { 1, 3, 0, 2 };
  for (int i = 0; i < 4; i++)
      shuffle_index_param->add_new_index(shuffle_index[i]);
  ShuffleIndexLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  }

}  // namespace caffe
