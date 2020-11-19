#include <cvdnn_core/utils.h>

namespace cvdnn {

bool GetNetParams(ros::NodeHandle& nh, NetParams& netParams) {
    if (!nh.getParam("model_path", netParams.mModel)) {
        return false;
    }
    nh.param<std::string>("config_path", netParams.mConfig, "");
    nh.param<std::string>("framework", netParams.mFramework, "caffe");
    nh.param<int>("backend_id", netParams.mBackendId, cv::dnn::DNN_BACKEND_DEFAULT);
    nh.param<int>("target_id", netParams.mTargetId, cv::dnn::DNN_TARGET_CPU);
    return true;
}

bool ReadNet(cv::dnn::Net& net,
             const NetParams& netParams) {
    if (netParams.mModel.empty()) {
        return false;
    }
    net = cv::dnn::readNet(netParams.mModel, netParams.mConfig, netParams.mFramework);
    net.setPreferableBackend(netParams.mBackendId);
    net.setPreferableTarget(netParams.mTargetId);
    return true;
}


cv::Mat Forward(cv::dnn::Net& net,
                const cv::Mat& frame,
                double scale,
                const cv::Size& size,
                const cv::Scalar& mean,
                bool swapRB) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, scale, size, mean, swapRB, false);
    net.setInput(blob);
    return net.forward();
}

}