#include <ros/ros.h>
#include <opencv2/dnn.hpp>

#include <string>

namespace cvdnn {

struct NetParams {
    NetParams(const std::string& model = "",
              const std::string& config = "",
              const std::string& framework = "caffe",
              int backendId = cv::dnn::DNN_BACKEND_DEFAULT,
              int targetId = cv::dnn::DNN_TARGET_CPU)
    : model_(model), config_(config), framework_(framework),
    backend_id_(backendId), target_id_(targetId) {};
    std::string model_;
    std::string config_;
    std::string framework_;
    int backend_id_;
    int target_id_;
};

bool GetNetParams(ros::NodeHandle& nh, NetParams& netParams);

bool ReadNet(cv::dnn::Net& net, const NetParams& netParams);

cv::Mat Forward(cv::dnn::Net& net,
                const cv::Mat& frame,
                double scale,
                const cv::Size& size,
                const cv::Scalar& mean = cv::Scalar(),
                bool swapRB = false);
}