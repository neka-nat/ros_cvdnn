#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/Classification2D.h>
#include <cvdnn_core/utils.h>
#include <fstream>

using namespace cvdnn;

namespace {

std::vector<std::string> readClassNames(const std::string& filename)
{
    std::vector<std::string> classNames;
    std::ifstream fp(filename.c_str());
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }
    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name.substr(name.find(' ') + 1));
    }
    fp.close();
    return classNames;
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
        mPub = mNh.advertise<vision_msgs::Classification2D>("classification", 5);
        mPubOvl = mNh.advertise<sensor_msgs::Image>("overlay", 2);
        if (!classNameTxt.empty()) mClassNames = readClassNames(classNameTxt);
    }

    void ImageCallback(const sensor_msgs::ImageConstPtr& data) {
        cv::Mat image;
        try {
            image = cv_bridge::toCvCopy(data, sensor_msgs::image_encodings::BGR8)->image;
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat prob = Forward(mNet, image, mScale, mSize, cv::Scalar(104, 117, 123));

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        vision_msgs::Classification2D msg;
        vision_msgs::ObjectHypothesis obj;
        obj.id = classId;
        obj.score = confidence;
        msg.results.push_back(obj);
        msg.header.stamp = ros::Time::now();

        std::string label = cv::format("%s: %.4f", (mClassNames.empty() ? cv::format("Class #%d", classId).c_str() :
                                                    mClassNames[classId].c_str()),
                                       confidence);
        cv::putText(image, label, cv::Point(0, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0));
        sensor_msgs::ImagePtr overlay = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
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
};

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cvdnn_classification");
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
    privateNh.param<int>("input_width", inputWidth, 224);
    privateNh.param<int>("input_height", inputHeight, 224);

    ImageProcessor ip(nh, net, inputName, scale, cv::Size(inputWidth, inputHeight), classNameTxt);
    ros::Rate rate(frq);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}