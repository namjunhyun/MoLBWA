// ROS2 port of the oCamS-1MGN-U stereo+IMU driver (originally ROS1 People_counter/oCamS.cpp).
// Publishes rectified mono8 stereo images (camera/left, camera/right) and IMU samples (imu),
// matching the topic names expected by orbslam3_ros2's stereo-inertial node.

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <opencv2/opencv.hpp>

#include "withrobot_camera.hpp"
#include "myahrs_plus.hpp"

using namespace std::chrono_literals;

namespace
{

class StereoCamera
{
public:
  Withrobot::Camera * camera;
  Withrobot::camera_format camFormat;
  int width_;
  int height_;

  StereoCamera(int resolution, double frame_rate)
  : camera(nullptr)
  {
    std::string dev_path = enum_dev_list();
    if (dev_path.empty()) {
      throw std::runtime_error("oCamS-1CGN-U/1MGN-U device not found");
    }

    camera = new Withrobot::Camera(dev_path.c_str());

    switch (resolution) {
      case 0: width_ = 1280; height_ = 960; break;
      case 1: width_ = 1280; height_ = 720; break;
      case 2: width_ = 640; height_ = 480; break;
      case 3: width_ = 640; height_ = 360; break;
      case 4: width_ = 320; height_ = 240; break;
      default: width_ = 640; height_ = 480; break;
    }

    camera->set_format(
      width_, height_, Withrobot::fourcc_to_pixformat('Y', 'U', 'Y', 'V'), 1,
      static_cast<unsigned int>(frame_rate));

    camera->get_current_format(camFormat);
    camFormat.print();

    camera->start();
  }

  ~StereoCamera()
  {
    camera->stop();
    delete camera;
  }

  static std::string enum_dev_list()
  {
    std::vector<Withrobot::usb_device_info> dev_list;
    int dev_num = Withrobot::get_usb_device_info_list(dev_list);
    if (dev_num < 1) {
      return "";
    }
    for (const auto & dev : dev_list) {
      if (dev.product == "oCamS-1CGN-U" || dev.product == "oCamS-1MGN-U") {
        return dev.dev_node;
      }
    }
    return "";
  }

  void uvc_control(int exposure, int gain, int blue, int red, bool ae)
  {
    camera->set_control("Exposure (Absolute)", exposure);
    camera->set_control("Gain", gain);
    camera->set_control("White Balance Blue Component", blue);
    camera->set_control("White Balance Red Component", red);
    camera->set_control("Exposure, Auto", ae ? 0x3 : 0x1);
  }

  bool getImages(cv::Mat & left_image, cv::Mat & right_image, uint32_t & time_stamp)
  {
    cv::Mat srcImg(cv::Size(camFormat.width, camFormat.height), CV_8UC2);
    cv::Mat dstImg[2];

    if (camera->get_frame(srcImg.data, camFormat.image_size, 1) == -1) {
      return false;
    }

    uint32_t ts;
    memcpy(&ts, srcImg.data, sizeof(ts));
    cv::split(srcImg, dstImg);

    time_stamp = ts;
    right_image = dstImg[0];
    left_image = dstImg[1];
    return true;
  }
};

}  // namespace

using WithrobotIMU::iMyAhrsPlus;
using WithrobotIMU::SensorData;

class OcamsStereoImuNode : public rclcpp::Node, public iMyAhrsPlus
{
public:
  OcamsStereoImuNode()
  : rclcpp::Node("ocams_stereo_imu_node"),
    iMyAhrsPlus(
      declare_and_get<std::string>("imu_port", "/dev/ttyACM0"),
      declare_and_get<int>("imu_baud_rate", 115200))
  {
    resolution_ = declare_and_get<int>("resolution", 2);
    frame_rate_ = declare_and_get<double>("frame_rate", 30.0);
    exposure_ = declare_and_get<int>("exposure", 100);
    gain_ = declare_and_get<int>("gain", 150);
    wb_blue_ = declare_and_get<int>("wb_blue", 200);
    wb_red_ = declare_and_get<int>("wb_red", 160);
    autoexposure_ = declare_and_get<bool>("auto_exposure", false);
    left_frame_id_ = declare_and_get<std::string>("left_frame_id", "left_camera");
    right_frame_id_ = declare_and_get<std::string>("right_frame_id", "right_camera");
    imu_frame_id_ = declare_and_get<std::string>("imu_frame_id", "imu_link");
    std::string imu_mode = declare_and_get<std::string>("imu_mode", "AMGQUA");
    std::string calib_dir = declare_and_get<std::string>("calib_dir", "");

    left_pub_ = create_publisher<sensor_msgs::msg::Image>("camera/left", 100);
    right_pub_ = create_publisher<sensor_msgs::msg::Image>("camera/right", 100);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("imu", 1000);

    ocams_ = std::make_unique<StereoCamera>(resolution_, frame_rate_);
    ocams_->uvc_control(exposure_, gain_, wb_blue_, wb_red_, autoexposure_);
    RCLCPP_INFO(get_logger(), "oCamS camera initialized (%dx%d)", ocams_->width_, ocams_->height_);

    setupRectification(calib_dir);

    if (!iMyAhrsPlus::start()) {
      RCLCPP_ERROR(get_logger(), "Failed to open IMU serial port. Check udev rules / permissions.");
    } else {
      // The device may already be streaming in the requested mode (persisted from a
      // prior session) and not ACK a redundant mode-set command, so a failure here
      // isn't necessarily fatal -- data will still flow if it's already correct.
      if (!iMyAhrsPlus::cmd_data_format(imu_mode.c_str(), 1000)) {
        RCLCPP_WARN(
          get_logger(),
          "IMU did not ACK data-format command (%s); assuming it is already streaming "
          "in this mode. If /imu stays silent, power-cycle the sensor.",
          imu_mode.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "IMU initialized: %s", imu_mode.c_str());
      }
    }

    capture_thread_ = std::thread(&OcamsStereoImuNode::captureLoop, this);
  }

  ~OcamsStereoImuNode() override
  {
    running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
  }

private:
  template<typename T>
  T declare_and_get(const std::string & name, T default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, default_value);
    }
    return get_parameter(name).get_value<T>();
  }

  void setupRectification(const std::string & calib_dir)
  {
    std::string base_dir = calib_dir.empty() ?
      (ament_index_cpp::get_package_share_directory("ocams_ros2") + "/config") : calib_dir;
    std::string left_path = base_dir + "/left_opencv.yaml";
    std::string right_path = base_dir + "/right_opencv.yaml";

    cv::FileStorage fsl(left_path, cv::FileStorage::READ);
    cv::FileStorage fsr(right_path, cv::FileStorage::READ);
    if (!fsl.isOpened() || !fsr.isOpened()) {
      RCLCPP_WARN(
        get_logger(),
        "Could not open calibration files (%s, %s); publishing UNRECTIFIED images",
        left_path.c_str(), right_path.c_str());
      rectify_ = false;
      return;
    }

    cv::Mat M1, D1, R1, P1, M2, D2, R2, P2;
    fsl["camera_matrix"] >> M1;
    fsl["distortion_coefficients"] >> D1;
    fsl["rectification_matrix"] >> R1;
    fsl["projection_matrix"] >> P1;
    fsr["camera_matrix"] >> M2;
    fsr["distortion_coefficients"] >> D2;
    fsr["rectification_matrix"] >> R2;
    fsr["projection_matrix"] >> P2;

    cv::Size img_size(ocams_->width_, ocams_->height_);
    cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_32FC1, map1x_, map1y_);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_32FC1, map2x_, map2y_);
    rectify_ = true;
    RCLCPP_INFO(get_logger(), "Loaded stereo rectification maps");
  }

  void captureLoop()
  {
    cv::Mat left_raw, right_raw;
    cv::Mat left_gray, right_gray;
    cv::Mat left_rect, right_rect;
    uint32_t dev_ts;

    while (running_ && rclcpp::ok()) {
      if (!ocams_->getImages(left_raw, right_raw, dev_ts)) {
        std::this_thread::sleep_for(1ms);
        continue;
      }

      rclcpp::Time stamp = now();

      cv::cvtColor(left_raw, left_gray, cv::COLOR_BayerGR2GRAY);
      cv::cvtColor(right_raw, right_gray, cv::COLOR_BayerGR2GRAY);

      if (rectify_) {
        cv::remap(left_gray, left_rect, map1x_, map1y_, cv::INTER_LINEAR);
        cv::remap(right_gray, right_rect, map2x_, map2y_, cv::INTER_LINEAR);
      } else {
        left_rect = left_gray;
        right_rect = right_gray;
      }

      left_pub_->publish(*toImageMsg(left_rect, stamp, left_frame_id_));
      right_pub_->publish(*toImageMsg(right_rect, stamp, right_frame_id_));
    }
  }

  static sensor_msgs::msg::Image::UniquePtr toImageMsg(
    const cv::Mat & img, const rclcpp::Time & stamp, const std::string & frame_id)
  {
    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp = stamp;
    msg->header.frame_id = frame_id;
    msg->height = img.rows;
    msg->width = img.cols;
    msg->encoding = "mono8";
    msg->is_bigendian = false;
    msg->step = img.cols;
    msg->data.assign(img.data, img.data + img.total() * img.elemSize());
    return msg;
  }

  // iMyAhrsPlus callback (fires from the SDK's internal receiver thread)
  void OnSensorData(int /*sensor_id*/, SensorData data) override
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);

    auto msg = sensor_msgs::msg::Imu();
    msg.header.stamp = now();
    msg.header.frame_id = imu_frame_id_;

    // The AMGQUA ASCII sentence sends pre-scaled fixed-point integers as decimal text
    // (verified empirically against raw serial capture): quaternion components are
    // Q14 fixed-point (divide by 16384 -> unit-magnitude quaternion, confirmed),
    // accel raw/100 already yields m/s^2 directly (confirmed against resting 1g
    // magnitude), gyro raw/900 already yields rad/s directly. These match the
    // divisors used by Withrobot's own ROS1 example driver.
    msg.orientation.x = static_cast<double>(data.quaternion.x) / 16384.0;
    msg.orientation.y = static_cast<double>(data.quaternion.y) / 16384.0;
    msg.orientation.z = static_cast<double>(data.quaternion.z) / 16384.0;
    msg.orientation.w = static_cast<double>(data.quaternion.w) / 16384.0;

    msg.linear_acceleration.x = static_cast<double>(data.imu.ax) / 100.0;
    msg.linear_acceleration.y = static_cast<double>(data.imu.ay) / 100.0;
    msg.linear_acceleration.z = static_cast<double>(data.imu.az) / 100.0;

    msg.angular_velocity.x = static_cast<double>(data.imu.gx) / 900.0;
    msg.angular_velocity.y = static_cast<double>(data.imu.gy) / 900.0;
    msg.angular_velocity.z = static_cast<double>(data.imu.gz) / 900.0;

    msg.linear_acceleration_covariance[0] = msg.linear_acceleration_covariance[4] =
      msg.linear_acceleration_covariance[8] = 0.05;
    msg.angular_velocity_covariance[0] = msg.angular_velocity_covariance[4] =
      msg.angular_velocity_covariance[8] = 0.025;
    msg.orientation_covariance[0] = msg.orientation_covariance[4] =
      msg.orientation_covariance[8] = 0.1;

    imu_pub_->publish(msg);
  }

  void OnAttributeChange(int /*sensor_id*/, std::string attribute_name, std::string value) override
  {
    RCLCPP_DEBUG(get_logger(), "IMU attribute changed: %s = %s", attribute_name.c_str(), value.c_str());
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

  std::unique_ptr<StereoCamera> ocams_;
  std::thread capture_thread_;
  std::atomic<bool> running_{true};
  std::mutex imu_mutex_;

  int resolution_;
  double frame_rate_;
  int exposure_, gain_, wb_blue_, wb_red_;
  bool autoexposure_;
  bool rectify_{false};
  std::string left_frame_id_, right_frame_id_, imu_frame_id_;
  cv::Mat map1x_, map1y_, map2x_, map2y_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OcamsStereoImuNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
