#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
public:
    explicit DataTransformer(const TransformationParameter& param, Phase phase);
    virtual ~DataTransformer() {}

    /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
    void InitRand();

    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
    void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
    void Transform(const vector<Datum> & datum_vector,
                   Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
    void Transform(const vector<cv::Mat> & mat_vector,
                   Blob<Dtype>* transformed_blob);

    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
    void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
#endif  // USE_OPENCV

    /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
    void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

    /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
    vector<int> InferBlobShape(const Datum& datum);
    /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
    vector<int> InferBlobShape(const vector<Datum> & datum_vector);
    /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
#ifdef USE_OPENCV
    vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
    /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
    vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data. This TransformVariedSizeDatum()
   * function is different with the original Transform() function that it
   * can process varied height and width of images in datum. For example,
   * images can be resized before saving them into a database.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
    void TransformVariedSizeDatum(const Datum& datum, Blob<Dtype>* transformed_blob);

    /**
   * @brief Similar to tranformations in TransformVariedSizeDatum, but output a blob
   * with 10 views for testing.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   * set_cpu_data() is used.
   */
    void TransformVariedSizeTestDatum(const Datum& datum, Blob<Dtype>* transformed_blob, const int num_test_views=10);

    /**
   * @brief Similar to transformations in TransformVariedSizeDatum, the function is applied to process rgb and
   * flow data.
   *
   * @param datum Input datum.
   * @param is_flow Wether input datum contains rgb or flow.
   * @param transformed_blob The tranformed data.
   */
    void TransformVariedSizeTwostreamDatum(const Datum& rgb_datum, const Datum& flow_datum,
                                           Blob<Dtype>* transformed_rgb_blob, Blob<Dtype>* transformed_flow_blob);

    /**
     * @brief Similar to transformations in TransformVariedSizeTestDatum, the function is applied to process rgb and
     * flow data. It performs data transform for TEST phase to extract 1-view or 10-view features.
     *
     * @param rgb_datum
     * @param flow_datum
     * @param transformed_rgb_blob
     * @param transformed_flow_blob
     * @param num_test_views
     */
    void TransformVariedSizeTwostreamTestDatum(const Datum& rgb_datum, const Datum& flow_datum,
                                               Blob<Dtype>* transformed_rgb_blob, Blob<Dtype>* transformed_flow_blob, const int num_test_views);

protected:
    /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
    virtual int Rand(int n);

    // protected functions
    void Transform(const Datum& datum, Dtype* transformed_data);

    void TransformVariedSizeDatum(const Datum& datum, Dtype* transformed_data);

    void TransformVariedSizeTestDatum(const Datum& datum, Dtype* transformed_data, const int num_test_views=10);

    void TransformVariedSizeTwostreamDatum(const Datum& rgb_datum, const Datum& flow_datum,
                                           Dtype* transformed_rgb_data, Dtype* transformed_flow_data);

    void TransformVariedSizeTwostreamTestDatum(const Datum& rgb_datum, const Datum& flow_datum,
                                               Dtype* transformed_rgb_data, Dtype* transformed_flow_data, const int num_test_views=10);

    // Tranformation parameters
    TransformationParameter param_;

    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    Blob<Dtype> data_mean_;
    vector<Dtype> mean_values_;

    vector<float> custom_scale_ratios_;
    int max_distort_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
