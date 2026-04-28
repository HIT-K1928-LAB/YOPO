#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <limits>
#include <memory>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "sensor_simulator.cuh"
#include <chrono>
#include "esdf_map.hpp"
#include "maps.hpp"
#include <pcl/filters/crop_box.h>

using namespace raycast;

namespace {
Eigen::Vector3f loadVector3(const YAML::Node &node, const Eigen::Vector3f &fallback) {
    if (!node || !node.IsSequence() || node.size() != 3) {
        return fallback;
    }
    return Eigen::Vector3f(node[0].as<float>(), node[1].as<float>(), node[2].as<float>());
}
}

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // 读取camera参数
        camera = new CameraParams();
        camera->fx = config["camera"]["fx"].as<float>();
        camera->fy = config["camera"]["fy"].as<float>();
        camera->cx = config["camera"]["cx"].as<float>();
        camera->cy = config["camera"]["cy"].as<float>();
        camera->image_width = config["camera"]["image_width"].as<int>();
        camera->image_height = config["camera"]["image_height"].as<int>();
        camera->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        camera->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
        quat_bc = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

        // 读取lidar参数
        lidar = new LidarParams();
        lidar->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();
        depth_pub_duration = ros::Duration(1 / depth_fps);
        lidar_pub_duration = ros::Duration(1 / lidar_fps);

        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = config["odom_topic"].as<std::string>();
        std::string depth_topic = config["depth_topic"].as<std::string>();
        std::string lidar_topic = config["lidar_topic"].as<std::string>();
        const float mock_map_voxel_resolution =
            config["mock_map_voxel_resolution"] ? config["mock_map_voxel_resolution"].as<float>() : 0.5f;
        esdf_enabled_ = config["build_esdf"] ? config["build_esdf"].as<bool>() : true;
        collision_check_ = config["enable_collision_check"] ? config["enable_collision_check"].as<bool>() : true;
        collision_radius_ = config["collision_radius"] ? config["collision_radius"].as<float>() : 0.3f;
        std::string collision_topic = config["collision_topic"] ? config["collision_topic"].as<std::string>() : "/sim/collision";
        std::string clearance_topic = config["clearance_topic"] ? config["clearance_topic"].as<std::string>() : "/sim/collision_clearance";

        EsdfMapConfig esdf_config;
        esdf_config.resolution = config["esdf_resolution"] ? config["esdf_resolution"].as<float>() : 0.2f;
        esdf_config.expand_min = loadVector3(config["esdf_expand_min"], Eigen::Vector3f(0.0f, 0.0f, 0.2f));
        esdf_config.expand_max = loadVector3(config["esdf_expand_max"], Eigen::Vector3f(0.0f, 0.0f, 6.0f));

        // 读取地图参数
        bool use_random_map = config["random_map"].as<bool>();
        float resolution = config["resolution"].as<float>();
        int occupy_threshold = config["occupy_threshold"].as<int>();
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("mock_map", 1);
        voxel_pcl_pub_ = nh.advertise<sensor_msgs::PointCloud2>("mock_map_voxel", 1);
        int seed = config["seed"].as<int>();
        double scale = 1 / resolution;
        int actual_x = config["x_length"].as<int>();
        int actual_y = config["y_length"].as<int>();
        int actual_z = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();

        int sizeX = actual_x * scale;
        int sizeY = actual_y * scale;
        int sizeZ = actual_z * scale;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_random_map) {
            printf("1.Generate Random Map... \n");
            mocka::Maps::BasicInfo info;
            info.sizeX      = sizeX;
            info.sizeY      = sizeY;
            info.sizeZ      = sizeZ;
            info.seed       = seed;
            info.scale      = scale;
            info.cloud      = cloud;

            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(type);
        }
        else {
            printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
            if (pcl::io::loadPLYFile(ply_file, *cloud) == -1) {
                PCL_ERROR("Couldn't read PLY file \n");
            }
        }

        // 裁剪点云
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(- actual_x / 2.0f, -actual_y / 2.0f, 0, 1.0f));
        crop.setMax(Eigen::Vector4f(actual_x / 2.0f, actual_y / 2.0f, actual_z / 2.0f, 1.0f));

        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(new pcl::PointCloud<pcl::PointXYZ>());
        crop.filter(*cropped);
        cloud = cropped;


        pcl::toROSMsg(*cloud, output_raw_);
        output_raw_.header.frame_id = "world";

        pcl::PointCloud<pcl::PointXYZ>::Ptr mock_map_voxel(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(mock_map_voxel_resolution,
                                 mock_map_voxel_resolution,
                                 mock_map_voxel_resolution);
        voxel_filter.filter(*mock_map_voxel);
        pcl::toROSMsg(*mock_map_voxel, output_voxel_);
        output_voxel_.header.frame_id = "world";
        ROS_INFO("Publishing mock_map_voxel with %.2f m resolution: %zu / %zu points",
                 mock_map_voxel_resolution,
                 mock_map_voxel->points.size(),
                 cloud->points.size());

        std::cout<<"Pointloud size:"<<cloud->points.size()<<std::endl;
        printf("2.Mapping... \n");
        grid_map = new GridMap(cloud, resolution, occupy_threshold);

        if (esdf_enabled_) {
            printf("3.Building ESDF... \n");
            esdf_map_ = std::make_unique<EsdfMap>(cloud, esdf_config);
        }
        if (!esdf_map_) {
            esdf_enabled_ = false;
            collision_check_ = false;
        }

        next_depth_pub_time = ros::Time::now();
        next_lidar_pub_time = ros::Time::now();

        // ROS
        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        if (esdf_enabled_) {
            collision_pub_ = nh_.advertise<std_msgs::Bool>(collision_topic, 1);
            clearance_pub_ = nh_.advertise<std_msgs::Float32>(clearance_topic, 1);
        }
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this, ros::TransportHints().tcpNoDelay());
        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);

        printf("4.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);

    void renderDepthCallback(const ros::Time stamp);

    void renderLidarCallback(const ros::Time stamp);

    void timerMapCallback(const ros::TimerEvent &);

    void publishCollisionState();

private:
    bool render_depth{false};
    bool render_lidar{false};
    bool esdf_enabled_{false};
    bool collision_check_{false};
    bool collision_state_initialized_{false};
    bool last_collision_state_{false};
    Eigen::Quaternionf quat;
    Eigen::Quaternionf quat_bc, quat_wc;
    Eigen::Vector3f pos;

    CameraParams* camera;
    LidarParams* lidar;
    GridMap* grid_map;
    std::unique_ptr<EsdfMap> esdf_map_;
    sensor_msgs::PointCloud2 output_raw_, output_voxel_;

    ros::NodeHandle nh_;
    ros::Publisher image_pub_, point_cloud_pub_;
    ros::Publisher pcl_pub, voxel_pcl_pub_, collision_pub_, clearance_pub_;
    ros::Subscriber odom_sub_;
    ros::Timer timer_depth_, timer_lidar_, timer_map_;

    ros::Time next_depth_pub_time, next_lidar_pub_time;
    ros::Duration depth_pub_duration, lidar_pub_duration;
    double depth_time{0.0}, lidar_time{0.0};
    int depth_count{0}, lidar_count{0};
    float collision_radius_{0.3f};
    float last_clearance_{std::numeric_limits<float>::quiet_NaN()};
    // mocka::Maps map;
};



void SensorSimulator::renderDepthCallback(const ros::Time stamp) {
    if (!render_depth)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(), pos.x(), pos.y(), pos.z());
    cv::Mat depth_image;
    renderDepthImage(grid_map, camera, T_wc, depth_image);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    depth_time += elapsed.count();
    depth_count++;
    // std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = stamp;
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);
}

void SensorSimulator::timerMapCallback(const ros::TimerEvent&) {
    if (pcl_pub.getNumSubscribers() > 0)
        pcl_pub.publish(output_raw_);
    if (voxel_pcl_pub_.getNumSubscribers() > 0)
        voxel_pcl_pub_.publish(output_voxel_);
}

void SensorSimulator::publishCollisionState() {
    if (!collision_check_ || !esdf_map_) {
        return;
    }

    const float clearance = esdf_map_->queryDistance(pos);
    const bool in_collision = clearance <= collision_radius_;

    std_msgs::Float32 clearance_msg;
    clearance_msg.data = clearance;
    clearance_pub_.publish(clearance_msg);

    std_msgs::Bool collision_msg;
    collision_msg.data = in_collision;
    collision_pub_.publish(collision_msg);

    if (!collision_state_initialized_ || in_collision != last_collision_state_) {
        if (in_collision) {
            ROS_ERROR("Collision detected! clearance=%.3f m radius=%.3f m", clearance, collision_radius_);
        } else {
            ROS_INFO("Collision cleared. clearance=%.3f m", clearance);
        }
    }

    collision_state_initialized_ = true;
    last_collision_state_ = in_collision;
    last_clearance_ = clearance;
}

void SensorSimulator::renderLidarCallback(const ros::Time stamp) {
    if (!render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time += elapsed.count();
    lidar_count++;
    // std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;
    quat_wc = quat * quat_bc;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    publishCollisionState();

    ros::Time tnow = ros::Time::now();

    // 避免仿真odom消息中断，导致时间差太大
    if (fabs((tnow - next_depth_pub_time).toSec()) > 10 * depth_pub_duration.toSec())
        next_depth_pub_time = tnow;
    if (fabs((tnow - next_lidar_pub_time).toSec()) > 10 * lidar_pub_duration.toSec())
        next_lidar_pub_time = tnow;

    if (tnow >= next_depth_pub_time){
        next_depth_pub_time += depth_pub_duration;
        renderDepthCallback(msg->header.stamp);
    }
    if (tnow >= next_lidar_pub_time){
        next_lidar_pub_time += lidar_pub_duration;
        renderLidarCallback(msg->header.stamp);
    }
    ros::Duration render_duration = ros::Time::now() - tnow;
    if (render_duration > depth_pub_duration || render_duration > lidar_pub_duration){
        // Performance reference: should take < 1 ms on 3060 GPU & Ubuntu 20.04
        ROS_WARN("Current Rendering time: %.2f ms, delay too much!", 1000 * render_duration.toSec());
        std::cout << "Average Depth Rendering time: " << (depth_time / (depth_count + 1e-8)) * 1000 << " ms" << std::endl;
        std::cout << "Average Lidar Rendering time: " << (lidar_time / (lidar_count + 1e-8)) * 1000 << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    SensorSimulator sensor_simulator(nh);
    return 0;
}
