#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <GoSdk/GoSdk.h>

#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <mutex>
#include <filesystem>

#include "gocator_msgs/srv/set_laser_state.hpp"

#define INVALID_RANGE_16BIT     ((k16s)0x8000)
#define DOUBLE_MAX_VALUE        ((k64f)1.7976931348623157e+308)
#define INVALID_RANGE_DOUBLE    ((k64f)-DOUBLE_MAX_VALUE)

#define NM_TO_MM(VALUE) (((k64f)(VALUE)) / 1000000.0)
#define UM_TO_MM(VALUE) (((k64f)(VALUE)) / 1000.0)

struct ProfilePoint
{
    double x = 0.0;      // mm, position along laser line
    double y = 0.0;      // mm, fixed to 0 for one 2D profile
    double z = INVALID_RANGE_DOUBLE;  // mm, height/range value
    uint8_t intensity = 0;
    bool valid = false;

    // Raw values received from Gocator before unit conversion.
    // For UNIFORM_PROFILE:
    //   raw_x = profile index
    //   raw_z = 16-bit range value
    // For PROFILE_POINT_CLOUD:
    //   raw_x = data[idx].x
    //   raw_z = data[idx].y
    int32_t raw_x = 0;
    int32_t raw_z = 0;
};

class GocatorProfileDriver : public rclcpp::Node
{
public:
    explicit GocatorProfileDriver();
    ~GocatorProfileDriver();

    bool init();
    void startThread();
    void run();
    void stop();

private:
    void processData(GoDataSet dataset);
    void publishProfile(const std::vector<ProfilePoint>& profile,
                        const GoStamp* stamp,
                        const std::string& profile_type,
                        uint32_t valid_count);

    void saveProfileToTxt(const std::vector<ProfilePoint>& profile,
                          const GoStamp* stamp,
                          const std::string& profile_type,
                          uint32_t valid_count);

    void handleSetLaserState(
        const std::shared_ptr<gocator_msgs::srv::SetLaserState::Request> request,
        const std::shared_ptr<gocator_msgs::srv::SetLaserState::Response> response);

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr profile_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Service<gocator_msgs::srv::SetLaserState>::SharedPtr laser_service_;

    kAssembly api_ = kNULL;
    GoSystem system_ = kNULL;
    GoSensor sensor_ = kNULL;
    kIpAddress ip_address_;

    std::string sensor_ip_;
    int receive_timeout_;
    bool initial_laser_state_;
    bool set_profile_mode_;
    bool publish_invalid_points_;
    std::string output_topic_;
    std::string frame_id_;
    std::string laser_service_name_;

    // Debug txt logging.
    bool save_profile_txt_;
    std::string profile_txt_path_;
    int max_debug_frames_;
    int image_width_for_mapping_;
    uint64_t saved_frame_count_ = 0;
    std::mutex txt_mutex_;

    std::thread run_thread_;
    std::atomic<bool> running_{false};
};

GocatorProfileDriver::GocatorProfileDriver()
    : Node("gocator_profile_driver")
{
    declare_parameter<std::string>("sensor_ip", "192.168.1.10");
    declare_parameter<int>("receive_timeout", 20000000);
    declare_parameter<bool>("initial_laser_state", false);
    declare_parameter<bool>("set_profile_mode", true);
    declare_parameter<bool>("publish_invalid_points", false);
    declare_parameter<std::string>("output_topic", "/gocator/profile_raw_mm");
    declare_parameter<std::string>("frame_id", "gocator_sensor");
    declare_parameter<std::string>("laser_service_name", "/gocator/set_laser_state");

    // Debug parameters.
    // save_profile_txt:
    //   true  = append every received profile to a txt file.
    //   false = do not write txt.
    // profile_txt_path:
    //   output txt path.
    // max_debug_frames:
    //   <=0 means unlimited; >0 saves only the first N frames.
    // image_width_for_mapping:
    //   If set to the video image width, an estimated image column u is also saved.
    //   0 means disabled.
    declare_parameter<bool>("save_profile_txt", false);
    declare_parameter<std::string>("profile_txt_path", "/tmp/gocator_profile_debug.txt");
    declare_parameter<int>("max_debug_frames", 20);
    declare_parameter<int>("image_width_for_mapping", 0);

    get_parameter("sensor_ip", sensor_ip_);
    get_parameter("receive_timeout", receive_timeout_);
    get_parameter("initial_laser_state", initial_laser_state_);
    get_parameter("set_profile_mode", set_profile_mode_);
    get_parameter("publish_invalid_points", publish_invalid_points_);
    get_parameter("output_topic", output_topic_);
    get_parameter("frame_id", frame_id_);
    get_parameter("laser_service_name", laser_service_name_);
    get_parameter("save_profile_txt", save_profile_txt_);
    get_parameter("profile_txt_path", profile_txt_path_);
    get_parameter("max_debug_frames", max_debug_frames_);
    get_parameter("image_width_for_mapping", image_width_for_mapping_);

    profile_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    laser_service_ = create_service<gocator_msgs::srv::SetLaserState>(
        laser_service_name_,
        std::bind(&GocatorProfileDriver::handleSetLaserState, this,
                  std::placeholders::_1, std::placeholders::_2));
}

GocatorProfileDriver::~GocatorProfileDriver()
{
    stop();
}

void GocatorProfileDriver::handleSetLaserState(
    const std::shared_ptr<gocator_msgs::srv::SetLaserState::Request> request,
    const std::shared_ptr<gocator_msgs::srv::SetLaserState::Response> response)
{
    if (sensor_ == kNULL) {
        response->success = false;
        response->message = "Sensor not initialized";
        RCLCPP_ERROR(get_logger(), "Attempted to set laser state before sensor initialization");
        return;
    }

    kStatus status = GoSensor_SetLaserState(sensor_, request->enable ? 1 : 0);
    if (status == kOK) {
        response->success = true;
        response->message = request->enable ? "Laser enabled" : "Laser disabled";
        RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    } else {
        response->success = false;
        response->message = "Failed to set laser state";
        RCLCPP_ERROR(get_logger(), "GoSensor_SetLaserState failed: %d", status);
    }
}

bool GocatorProfileDriver::init()
{
    kStatus status;

    if ((status = GoSdk_Construct(&api_)) != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSdk_Construct failed: %d", status);
        return false;
    }

    if ((status = GoSystem_Construct(&system_, kNULL)) != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSystem_Construct failed: %d", status);
        return false;
    }

    if (kIpAddress_Parse(&ip_address_, sensor_ip_.c_str()) != kOK) {
        RCLCPP_ERROR(get_logger(), "Invalid IP address: %s", sensor_ip_.c_str());
        return false;
    }

    if ((status = GoSystem_FindSensorByIpAddress(system_, &ip_address_, &sensor_)) != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSystem_FindSensorByIpAddress failed: %d", status);
        return false;
    }

    if ((status = GoSensor_Connect(sensor_)) != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSensor_Connect failed: %d", status);
        return false;
    }

    GoSetup setup = GoSensor_Setup(sensor_);
    if (setup == kNULL) {
        RCLCPP_ERROR(get_logger(), "GoSensor_Setup returned null");
        return false;
    }

    if (set_profile_mode_) {
        kStatus scanModeStatus = GoSetup_SetScanMode(setup, GO_MODE_PROFILE);
        if (scanModeStatus != kOK) {
            RCLCPP_WARN(get_logger(), "Failed to set scan mode to profile mode: %d", scanModeStatus);
        } else {
            RCLCPP_INFO(get_logger(), "Scan mode set to profile mode");
        }
    }

    status = GoSensor_SetLaserState(sensor_, initial_laser_state_ ? 1 : 0);
    if (status != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSensor_SetLaserState failed: %d", status);
        return false;
    }

    if ((status = GoSystem_EnableData(system_, kTRUE)) != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSystem_EnableData failed: %d", status);
        return false;
    }

    if ((status = GoSystem_Start(system_)) != kOK) {
        RCLCPP_ERROR(get_logger(), "GoSystem_Start failed: %d", status);
        return false;
    }

    RCLCPP_INFO(get_logger(), "Gocator profile driver initialized. Sensor IP: %s", sensor_ip_.c_str());
    return true;
}

void GocatorProfileDriver::startThread()
{
    running_ = true;
    run_thread_ = std::thread([this]() { run(); });
}

void GocatorProfileDriver::run()
{
    running_ = true;
    while (running_ && rclcpp::ok()) {
        GoDataSet dataset = kNULL;
        kStatus status = GoSystem_ReceiveData(system_, &dataset, receive_timeout_);

        if (status == kOK) {
            processData(dataset);
            GoDestroy(dataset);
        } else if (status != kERROR_TIMEOUT) {
            RCLCPP_ERROR(get_logger(), "Data receive error: %d", status);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void GocatorProfileDriver::stop()
{
    if (running_) {
        running_ = false;
        if (run_thread_.joinable()) {
            run_thread_.join();
        }
    }

    if (system_ != kNULL) {
        GoSystem_Stop(system_);
        GoDestroy(system_);
        system_ = kNULL;
    }

    if (api_ != kNULL) {
        GoDestroy(api_);
        api_ = kNULL;
    }

    RCLCPP_INFO(get_logger(), "Gocator profile driver stopped");
}

void GocatorProfileDriver::processData(GoDataSet dataset)
{
    GoStamp* stamp = nullptr;
    std::vector<ProfilePoint> profile;
    std::string profile_type;
    uint32_t valid_count = 0;
    
    // First pass: get stamp.
    for (k32u i = 0; i < GoDataSet_Count(dataset); ++i) {
        GoDataMsg dataObj = GoDataSet_At(dataset, i);
        if (GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_STAMP) {
            GoStampMsg stampMsg = dataObj;
            if (GoStampMsg_Count(stampMsg) > 0) {
                stamp = GoStampMsg_At(stampMsg, 0);
            }
        }
    }

    // Second pass: get profile and intensity.
    for (k32u i = 0; i < GoDataSet_Count(dataset); ++i) {
        GoDataMsg dataObj = GoDataSet_At(dataset, i);

        switch (GoDataMsg_Type(dataObj)) {
            case GO_DATA_MESSAGE_TYPE_UNIFORM_PROFILE: {
                GoResampledProfileMsg profileMsg = dataObj;
                if (GoResampledProfileMsg_Count(profileMsg) == 0) {
                    break;
                }

                kSize k = 0;
                kSize width = GoResampledProfileMsg_Width(profileMsg);
                k16s* data = GoResampledProfileMsg_At(profileMsg, k);
                double x_resolution = NM_TO_MM(GoResampledProfileMsg_XResolution(profileMsg));
                double z_resolution = NM_TO_MM(GoResampledProfileMsg_ZResolution(profileMsg));
                double x_offset = UM_TO_MM(GoResampledProfileMsg_XOffset(profileMsg));
                double z_offset = UM_TO_MM(GoResampledProfileMsg_ZOffset(profileMsg));

                static auto last_param_log = this->get_clock()->now();
                auto now = this->get_clock()->now();

                if ((now - last_param_log).seconds() >= 1.0)
                {
                    last_param_log = now;

                    RCLCPP_INFO(
                        get_logger(),
                        "[UniformProfile Params] "
                        "XOffset=%.6f mm, "
                        "ZOffset=%.6f mm, "
                        "XResolution=%.9f mm, "
                        "ZResolution=%.9f mm",
                        x_offset,
                        z_offset,
                        x_resolution,
                        z_resolution
                    );
                }


                profile.assign(width, ProfilePoint{});
                valid_count = 0;
                profile_type = "uniform_profile";

                for (kSize idx = 0; idx < width; ++idx) {
                    profile[idx].raw_x = static_cast<int32_t>(idx);
                    profile[idx].raw_z = static_cast<int32_t>(data[idx]);
                    profile[idx].x = x_offset + x_resolution * static_cast<double>(idx);
                    profile[idx].y = 0.0;
                    if (data[idx] != INVALID_RANGE_16BIT) {
                        profile[idx].z = z_offset + z_resolution * static_cast<double>(data[idx]);
                        profile[idx].valid = true;
                        valid_count++;
                    } else {
                        profile[idx].z = INVALID_RANGE_DOUBLE;
                        profile[idx].valid = false;
                    }
                }
                break;
            }

            case GO_DATA_MESSAGE_TYPE_PROFILE_POINT_CLOUD: {
                GoProfileMsg profileMsg = dataObj;
                if (GoProfileMsg_Count(profileMsg) == 0) {
                    break;
                }

                kSize k = 0;
                kSize width = GoProfileMsg_Width(profileMsg);
                kPoint16s* data = GoProfileMsg_At(profileMsg, k);
                double x_resolution = NM_TO_MM(GoProfileMsg_XResolution(profileMsg));
                double z_resolution = NM_TO_MM(GoProfileMsg_ZResolution(profileMsg));
                double x_offset = UM_TO_MM(GoProfileMsg_XOffset(profileMsg));
                double z_offset = UM_TO_MM(GoProfileMsg_ZOffset(profileMsg));

                profile.assign(width, ProfilePoint{});
                valid_count = 0;
                profile_type = "profile_point_cloud";

                for (kSize idx = 0; idx < width; ++idx) {
                    profile[idx].raw_x = static_cast<int32_t>(data[idx].x);
                    profile[idx].raw_z = static_cast<int32_t>(data[idx].y);
                    profile[idx].y = 0.0;
                    // GoProfileMsg encodes X and Z in kPoint16s::x/y.  A
                    // sample is usable only when both components are valid;
                    // accepting a valid X with INVALID_RANGE_16BIT in Z
                    // creates a plausible finite but physically false depth.
                    if (data[idx].x != INVALID_RANGE_16BIT &&
                        data[idx].y != INVALID_RANGE_16BIT) {
                        profile[idx].x = x_offset + x_resolution * static_cast<double>(data[idx].x);
                        profile[idx].z = z_offset + z_resolution * static_cast<double>(data[idx].y);
                        profile[idx].valid = true;
                        valid_count++;
                    } else {
                        profile[idx].x = INVALID_RANGE_DOUBLE;
                        profile[idx].z = INVALID_RANGE_DOUBLE;
                        profile[idx].valid = false;
                    }
                }
                break;
            }

            case GO_DATA_MESSAGE_TYPE_PROFILE_INTENSITY: {
                if (profile.empty()) {
                    break;
                }

                GoProfileIntensityMsg intensityMsg = dataObj;
                if (GoProfileIntensityMsg_Count(intensityMsg) == 0) {
                    break;
                }

                kSize k = 0;
                kSize width = GoProfileIntensityMsg_Width(intensityMsg);
                k8u* data = GoProfileIntensityMsg_At(intensityMsg, k);
                kSize copy_width = std::min<kSize>(width, profile.size());

                for (kSize idx = 0; idx < copy_width; ++idx) {
                    profile[idx].intensity = static_cast<uint8_t>(data[idx]);
                }
                break;
            }

            default:
                break;
        }
    }

    // Intensity and profile messages are not guaranteed to be ordered in a
    // GoDataSet.  Repeat the lightweight intensity pass after the geometry
    // has been decoded so intensity is retained even when it arrived first.
    if (!profile.empty()) {
        for (k32u i = 0; i < GoDataSet_Count(dataset); ++i) {
            GoDataMsg dataObj = GoDataSet_At(dataset, i);
            if (GoDataMsg_Type(dataObj) != GO_DATA_MESSAGE_TYPE_PROFILE_INTENSITY) {
                continue;
            }
            GoProfileIntensityMsg intensityMsg = dataObj;
            if (GoProfileIntensityMsg_Count(intensityMsg) == 0) {
                continue;
            }
            k8u* data = GoProfileIntensityMsg_At(intensityMsg, 0);
            const kSize copy_width = std::min<kSize>(
                GoProfileIntensityMsg_Width(intensityMsg), profile.size());
            for (kSize idx = 0; idx < copy_width; ++idx) {
                profile[idx].intensity = static_cast<uint8_t>(data[idx]);
            }
            break;
        }
    }

    if (!profile.empty()) {
        publishProfile(profile, stamp, profile_type, valid_count);
    }
}


void GocatorProfileDriver::saveProfileToTxt(const std::vector<ProfilePoint>& profile,
                                            const GoStamp* stamp,
                                            const std::string& profile_type,
                                            uint32_t valid_count)
{
    if (!save_profile_txt_) {
        return;
    }
    if (valid_count == 0) {
        return;
    }

    if (max_debug_frames_ > 0 && saved_frame_count_ >= static_cast<uint64_t>(max_debug_frames_)) {
        return;
    }

    std::lock_guard<std::mutex> lock(txt_mutex_);

    if (max_debug_frames_ > 0 && saved_frame_count_ >= static_cast<uint64_t>(max_debug_frames_)) {
        return;
    }

    std::filesystem::path out_path(profile_txt_path_);
    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }

    const bool file_exists = std::filesystem::exists(out_path);
    std::ofstream ofs(profile_txt_path_, std::ios::out | std::ios::app);
    if (!ofs.is_open()) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                              "Failed to open profile txt: %s", profile_txt_path_.c_str());
        return;
    }

    if (!file_exists) {
        ofs << "# Gocator profile debug data\n";
        ofs << "# Columns:\n";
        ofs << "# frame_saved_id, frame_index, gocator_timestamp, encoder, profile_type, "
               "profile_width, valid_count, profile_index, estimated_image_u, valid, "
               "raw_x, raw_z, x_mm, y_mm, z_mm, intensity\n";
        ofs << "# Notes:\n";
        ofs << "# - profile_index is the index in the returned profile array.\n";
        ofs << "# - estimated_image_u is computed only when image_width_for_mapping > 0.\n";
        ofs << "# - For UNIFORM_PROFILE, raw_x equals profile_index and raw_z is the 16-bit range value.\n";
        ofs << "# - For PROFILE_POINT_CLOUD, raw_x/raw_z are the original encoded x/y values from Gocator.\n";
        ofs << "# - x_mm/y_mm/z_mm are converted engineering-unit coordinates in the Gocator sensor frame.\n\n";
    }

    uint64_t frame_index = stamp ? stamp->frameIndex : 0;
    uint64_t gocator_timestamp = stamp ? stamp->timestamp : 0;
    int64_t encoder = stamp ? stamp->encoder : 0;

    ofs << std::fixed << std::setprecision(6);

    const size_t width = profile.size();
    for (size_t idx = 0; idx < width; ++idx) {
        const auto& p = profile[idx];
        if (!p.valid) {
            continue;
        }

        double estimated_image_u = -1.0;
        if (image_width_for_mapping_ > 0 && width > 1) {
            estimated_image_u =
                static_cast<double>(idx) *
                static_cast<double>(image_width_for_mapping_ - 1) /
                static_cast<double>(width - 1);
        }

        ofs << "saved_frame_count_" << ","
            << saved_frame_count_ << ","
            << "frame_index" << ","
            << frame_index << ","
            << "gocator_timestamp" << ","
            << gocator_timestamp << ","
            << "profile_type" << ","
            << profile_type << ","
            << "width" << ","
            << width << ","
            << "valid_count" << ","
            << valid_count << ","
            << "idx" << ","
            << idx << ","
            << "raw_x" << ","
            << p.raw_x << ","
            << "raw_z" << ","
            << p.raw_z << ",";

        if (p.valid) {
            ofs << "p.x" << ","
                << p.x << ","
                << "p.y" << ","
                << p.y << ","
                << "p.z" << ","
                << p.z << ",";
        } else {
            ofs << "nan,nan,nan,";
        }

        ofs << static_cast<int>(p.intensity) << "\n";
    }

    ofs << "\n";
    saved_frame_count_++;

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Saved profile txt frames: %llu/%d -> %s",
                         static_cast<unsigned long long>(saved_frame_count_),
                         max_debug_frames_,
                         profile_txt_path_.c_str());
}

void GocatorProfileDriver::publishProfile(const std::vector<ProfilePoint>& profile,
                                          const GoStamp* stamp,
                                          const std::string& profile_type,
                                          uint32_t valid_count)
{
    saveProfileToTxt(profile, stamp, profile_type, valid_count);

    size_t publish_count = 0;
    for (const auto& p : profile) {
        if (publish_invalid_points_ || p.valid) {
            publish_count++;
        }
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = this->now();
    cloud_msg.header.frame_id = frame_id_;
    cloud_msg.height = 1;
    cloud_msg.width = static_cast<uint32_t>(publish_count);
    cloud_msg.is_bigendian = false;
    cloud_msg.is_dense = !publish_invalid_points_;

    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2Fields(
        5,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::msg::PointField::UINT8,
        "index", 1, sensor_msgs::msg::PointField::UINT32);
    modifier.resize(publish_count);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_intensity(cloud_msg, "intensity");
    sensor_msgs::PointCloud2Iterator<uint32_t> iter_index(cloud_msg, "index");

    for (uint32_t idx = 0; idx < profile.size(); ++idx) {
        const auto& p = profile[idx];
        if (!publish_invalid_points_ && !p.valid) {
            continue;
        }

        *iter_x = p.valid ? static_cast<float>(p.x) : std::numeric_limits<float>::quiet_NaN();
        *iter_y = p.valid ? static_cast<float>(p.y) : std::numeric_limits<float>::quiet_NaN();
        *iter_z = p.valid ? static_cast<float>(p.z) : std::numeric_limits<float>::quiet_NaN();
        *iter_intensity = p.intensity;
        *iter_index = idx;

        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_intensity;
        ++iter_index;
    }

    profile_pub_->publish(cloud_msg);

    auto now = this->get_clock()->now();
    static auto last_log_time = now;
    if ((now - last_log_time).seconds() >= 1.0) {
        last_log_time = now;
        if (stamp) {
            RCLCPP_INFO(get_logger(),
                        "Published %s: points=%zu valid=%u frame=%llu timestamp=%llu encoder=%lld",
                        profile_type.c_str(), profile.size(), valid_count,
                        stamp->frameIndex, stamp->timestamp, stamp->encoder);
        } else {
            RCLCPP_INFO(get_logger(),
                        "Published %s: points=%zu valid=%u no_stamp",
                        profile_type.c_str(), profile.size(), valid_count);
        }
    }

    // FIXED: Gocator 装在机器人末端，TF 由 calib_tf_broadcaster 发布 (fanuc_flange->gocator_frame)
    // geometry_msgs::msg::TransformStamped transformStamped;
    // transformStamped.header.stamp = now;
    // transformStamped.header.frame_id = "world";
    // transformStamped.child_frame_id = "gocator_frame";
    // transformStamped.transform.translation.x = 0.0;
    // transformStamped.transform.translation.y = 0.0;
    // transformStamped.transform.translation.z = 0.0;
    // transformStamped.transform.rotation.x = 0.0;
    // transformStamped.transform.rotation.y = 0.0;
    // transformStamped.transform.rotation.z = 0.0;
    // transformStamped.transform.rotation.w = 1.0;
    // tf_broadcaster_->sendTransform(transformStamped);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<GocatorProfileDriver>();

    if (node->init()) {
        node->startThread();
        rclcpp::spin(node);
        node->stop();
    } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize Gocator profile driver");
    }

    rclcpp::shutdown();
    return 0;
}
