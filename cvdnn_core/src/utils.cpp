#include <cvdnn_core/utils.h>

namespace cvdnn {

bool GetNetParams(ros::NodeHandle& nh, NetParams& netParams) {
    if (!nh.getParam("model_path", netParams.model_)) {
        return false;
    }
    nh.param<std::string>("config_path", netParams.config_, "");
    nh.param<std::string>("framework", netParams.framework_, "caffe");
    nh.param<int>("backend_id", netParams.backend_id_, cv::dnn::DNN_BACKEND_DEFAULT);
    nh.param<int>("target_id", netParams.target_id_, cv::dnn::DNN_TARGET_CPU);
    return true;
}

bool ReadNet(cv::dnn::Net& net,
             const NetParams& netParams) {
    if (netParams.model_.empty()) {
        return false;
    }
    net = cv::dnn::readNet(netParams.model_, netParams.config_, netParams.framework_);
    net.setPreferableBackend(netParams.backend_id_);
    net.setPreferableTarget(netParams.target_id_);
    return true;
}


cv::Mat Forward(cv::dnn::Net& net,
                const cv::Mat& frame,
                double scale,
                const cv::Size& size,
                const cv::Scalar& mean,
                bool swapRB) {
    cv::Mat blob;
    const int inpWidth = frame.cols;
    const int inpHeight = frame.rows;
    cv::dnn::blobFromImage(frame, blob, scale, size, mean, swapRB, false);
    net.setInput(blob);
    return net.forward();
}

}