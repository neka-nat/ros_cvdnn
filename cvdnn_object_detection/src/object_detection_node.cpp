#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/Detection2DArray.h>
#include <cvdnn_core/utils.h>
#include <fstream>

using namespace cvdnn;

namespace {

std::vector<std::string> readClassNames(const std::string& filename)
{
    std::vector<std::string> classNamesVec;
    std::ifstream classNamesFile(filename.c_str());
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    return classNamesVec;
}

class ImageProcessor {
public:
    ImageProcessor(ros::NodeHandle& nh, cv::dnn::Net& net,
                   const std::string& inputName,
                   double scale,
                   const cv::Size& size,
                   const std::string& classNameTxt = "")
    : mNet(net), mNh(nh), mIt(nh), mScale(scale), mSize(size) {
        mSub = mIt.subscribe(inputName, 5, &ImageProcessor::ImageCallback, this);
        mPub = mNh.advertise<vision_msgs::Detection2DArray>("detections", 5);
        mPubOvl = mNh.advertise<sensor_msgs::Image>("overlay", 2);
        if (!classNameTxt.empty()) mClassNames = readClassNames(classNameTxt);
        std::vector<int> outLayers = mNet.getUnconnectedOutLayers();
        mOutLayerType = mNet.getLayer(outLayers[0])->type;
    }

    void ImageCallback(const sensor_msgs::ImageConstPtr& data) {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(data, sensor_msgs::image_encodings::BGR8)->image;
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat detectionMat = Forward(mNet, frame, mScale, mSize, cv::Scalar());

        vision_msgs::Detection2DArray msg;
        std::ostringstream ss;
        if (mOutLayerType == "DetectionOutput") {
            float* data = (float*)detectionMat.data;
            for (size_t i = 0; i < detectionMat.total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > mConfidenceThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2) {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    size_t objectClass = (int)(data[i + 1]) - 1;  // Skip 0th background class id.
                    vision_msgs::Detection2D detMsg;
                    detMsg.bbox.size_x = width;
                    detMsg.bbox.size_y = height;
                    detMsg.bbox.center.x = (left + right) / 2;
                    detMsg.bbox.center.y = (top + bottom) / 2;
                    detMsg.bbox.center.theta = 0.0;
                    vision_msgs::ObjectHypothesisWithPose hyp;
                    hyp.id = objectClass;
                    hyp.score = confidence;
                    msg.detections.push_back(detMsg);
                    cv::Rect object(left, bottom,
                                    width, height);
                    cv::rectangle(frame, object, cv::Scalar(0, 255, 0));
                    if (objectClass < mClassNames.size()) {
                        ss.str("");
                        ss << confidence;
                        std::string conf(ss.str());
                        std::string label = std::string(mClassNames[objectClass]) + ": " + conf;
                        int baseLine = 0;
                        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                        cv::rectangle(frame, cv::Rect(cv::Point(left, bottom),
                                             cv::Size(labelSize.width, labelSize.height + baseLine)),
                                     cv::Scalar(255, 255, 255), cv::FILLED);
                        cv::putText(frame, label, cv::Point(left, bottom + labelSize.height),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    }
                }
            }
        } else if (mOutLayerType == "Region") {
            for (int i = 0; i < detectionMat.rows; i++) {
                const int probability_index = 5;
                const int probability_size = detectionMat.cols - probability_index;
                float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
                size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
                float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
                if (confidence > mConfidenceThreshold)
                {
                    float x = detectionMat.at<float>(i, 0);
                    float y = detectionMat.at<float>(i, 1);
                    float width = detectionMat.at<float>(i, 2);
                    float height = detectionMat.at<float>(i, 3);
                    int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                    int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                    int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                    int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
                    vision_msgs::Detection2D detMsg;
                    detMsg.bbox.size_x = width * frame.cols;
                    detMsg.bbox.size_y = height * frame.rows;
                    detMsg.bbox.center.x = x * frame.cols;
                    detMsg.bbox.center.y = y * frame.rows;
                    detMsg.bbox.center.theta = 0.0;
                    vision_msgs::ObjectHypothesisWithPose hyp;
                    hyp.id = objectClass;
                    hyp.score = confidence;
                    msg.detections.push_back(detMsg);
                    cv::Rect object(xLeftBottom, yLeftBottom,
                                    xRightTop - xLeftBottom,
                                    yRightTop - yLeftBottom);
                    cv::rectangle(frame, object, cv::Scalar(0, 255, 0));
                    if (objectClass < mClassNames.size()) {
                        ss.str("");
                        ss << confidence;
                        std::string conf(ss.str());
                        std::string label = std::string(mClassNames[objectClass]) + ": " + conf;
                        int baseLine = 0;
                        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                        cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom),
                                             cv::Size(labelSize.width, labelSize.height + baseLine)),
                                     cv::Scalar(255, 255, 255), cv::FILLED);
                        cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom + labelSize.height),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    }
                }
            }
        }
        sensor_msgs::ImagePtr overlay = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        msg.header.stamp = ros::Time::now();
        mPub.publish(msg);
        mPubOvl.publish(overlay);
    }
private:
    cv::dnn::Net& mNet;
    ros::NodeHandle& mNh;
    image_transport::ImageTransport mIt;
    image_transport::Subscriber mSub;
    ros::Publisher mPub;
    ros::Publisher mPubOvl;
    double mScale;
    cv::Size mSize;
    std::vector<std::string> mClassNames;
    double mConfidenceThreshold;
    std::string mOutLayerType;
};

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cvdnn_object_detection");
    ros::NodeHandle nh;
    ros::NodeHandle privateNh("~");
    NetParams params;
    if (!GetNetParams(privateNh, params)) {
        ROS_ERROR("Invalid network params.");
        return 0;
    }
    cv::dnn::Net net;
    if (!ReadNet(net, params)) {
        ROS_ERROR("Can not read network.");
        return 0;
    }
    double frq;
    privateNh.param<double>("frequency", frq, 10);
    std::string inputName;
    privateNh.param<std::string>("input_topic_name", inputName, "image_in");
    std::string classNameTxt;
    privateNh.param<std::string>("class_names", classNameTxt, "");
    double scale;
    privateNh.param<double>("input_scale", scale, 1.0);
    int inputWidth;
    int inputHeight;
    privateNh.param<int>("input_width", inputWidth, 416);
    privateNh.param<int>("input_height", inputHeight, 416);


    ImageProcessor ip(nh, net, inputName, scale, cv::Size(inputWidth, inputHeight), classNameTxt);
    ros::Rate rate(frq);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}