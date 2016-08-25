#ifndef CAFFE_MULTIVIEW_AVERAGE_LAYER_HPP_
#define CAFFE_MULTIVIEW_AVERAGE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/video_test_data_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multiview average of a blob. This layer is mainly used in
 * TEST phase for evaluation. It is in use with VideoTestDataLayer (to extract
 * 10-view inputs). Passing an blob going through this layer before going to the
 * AccuracyLayer or SoftmaxLayer for final predictions.
 *
 */
template <typename Dtype>
class MultiviewAverageLayer : public Layer<Dtype> {
 public:
  /**
   * @param param,
   *     MultiviewAverageLayer does not need any parameters.
   *
   */
  explicit MultiviewAverageLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiviewAverage"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1;}

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating results from a computation
   *      in the network.
   * @param top output Blob vector (length 1)
   *   -# @f$ (N/10 \times C \times H \times W) @f$
   *      the averaged blob along first dimension of the input blob.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); i++) {
            if (propagate_down[i]) { NOT_IMPLEMENTED; }
        }
    }

  int height_, width_;
  int channels_;

};

}  // namespace caffe

#endif  // CAFFE_MULTIVIEW_AVERAGE_LAYER_HPP_
