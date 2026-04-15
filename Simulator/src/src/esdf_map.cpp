#include "esdf_map.hpp"

#include <pcl/common/common.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
constexpr float kLargeDistance = 1e12f;
}

EsdfMap::EsdfMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const EsdfMapConfig &config)
{
    if (!cloud || cloud->empty()) {
        throw std::runtime_error("ESDF build failed: input cloud is empty.");
    }

    resolution_ = config.resolution;

    pcl::PointXYZ min_point;
    pcl::PointXYZ max_point;
    pcl::getMinMax3D(*cloud, min_point, max_point);

    min_bound_ = Eigen::Vector3f(min_point.x, min_point.y, min_point.z) - config.expand_min;
    max_bound_ = Eigen::Vector3f(max_point.x, max_point.y, max_point.z) + config.expand_max;

    Eigen::Vector3f map_size = max_bound_ - min_bound_;
    grid_size_ = (map_size.array() / resolution_).ceil().cast<int>();
    grid_size_ = grid_size_.cwiseMax(Eigen::Vector3i::Ones() * 2);
    max_clearance_ = map_size.norm();

    const auto start_time = std::chrono::steady_clock::now();

    std::vector<uint8_t> occupancy(static_cast<size_t>(grid_size_.x()) * grid_size_.y() * grid_size_.z(), 0);
    buildOccupancy(cloud, occupancy);
    buildSignedDistanceField(occupancy);

    const double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "ESDF ready: res=" << resolution_
              << " size=(" << grid_size_.x() << ", " << grid_size_.y() << ", " << grid_size_.z() << ")"
              << " bounds=[(" << min_bound_.x() << ", " << min_bound_.y() << ", " << min_bound_.z() << ") -> ("
              << max_bound_.x() << ", " << max_bound_.y() << ", " << max_bound_.z() << ")]"
              << " build_time=" << elapsed_sec << " s" << std::endl;
}

int EsdfMap::flatten(int x, int y, int z) const
{
    return x * grid_size_.y() * grid_size_.z() + y * grid_size_.z() + z;
}

float EsdfMap::mirroredCoordinate(float coord, int length) const
{
    if (length <= 1) {
        return 0.0f;
    }

    const float max_index = static_cast<float>(length - 1);
    const float period = 2.0f * max_index;
    coord = std::fmod(coord, period);
    if (coord < 0.0f) {
        coord += period;
    }
    if (coord > max_index) {
        coord = period - coord;
    }
    return coord;
}

float EsdfMap::sampleVoxel(int x, int y, int z) const
{
    x = std::clamp(x, 0, grid_size_.x() - 1);
    y = std::clamp(y, 0, grid_size_.y() - 1);
    z = std::clamp(z, 0, grid_size_.z() - 1);
    return sdf_[flatten(x, y, z)];
}

float EsdfMap::trilinearSample(float gx, float gy, float gz) const
{
    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int z0 = static_cast<int>(std::floor(gz));
    const int x1 = std::min(x0 + 1, grid_size_.x() - 1);
    const int y1 = std::min(y0 + 1, grid_size_.y() - 1);
    const int z1 = std::min(z0 + 1, grid_size_.z() - 1);

    const float tx = gx - x0;
    const float ty = gy - y0;
    const float tz = gz - z0;

    const float c000 = sampleVoxel(x0, y0, z0);
    const float c001 = sampleVoxel(x0, y0, z1);
    const float c010 = sampleVoxel(x0, y1, z0);
    const float c011 = sampleVoxel(x0, y1, z1);
    const float c100 = sampleVoxel(x1, y0, z0);
    const float c101 = sampleVoxel(x1, y0, z1);
    const float c110 = sampleVoxel(x1, y1, z0);
    const float c111 = sampleVoxel(x1, y1, z1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

float EsdfMap::queryDistance(const Eigen::Vector3f &pos) const
{
    if (pos.z() <= 0.0f) {
        return pos.z() - resolution_;
    }

    float gz = (pos.z() - min_bound_.z()) / resolution_;
    if (gz < 0.0f) {
        return -resolution_;
    }
    if (gz > static_cast<float>(grid_size_.z() - 1)) {
        return max_clearance_;
    }

    float gx = (pos.x() - min_bound_.x()) / resolution_;
    float gy = (pos.y() - min_bound_.y()) / resolution_;
    gx = mirroredCoordinate(gx, grid_size_.x());
    gy = mirroredCoordinate(gy, grid_size_.y());
    return trilinearSample(gx, gy, gz);
}

void EsdfMap::buildOccupancy(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<uint8_t> &occupancy) const
{
    for (const auto &point : cloud->points) {
        const int x = static_cast<int>(std::floor((point.x - min_bound_.x()) / resolution_));
        const int y = static_cast<int>(std::floor((point.y - min_bound_.y()) / resolution_));
        const int z = static_cast<int>(std::floor((point.z - min_bound_.z()) / resolution_));
        if (x < 0 || x >= grid_size_.x() || y < 0 || y >= grid_size_.y() || z < 0 || z >= grid_size_.z()) {
            continue;
        }
        occupancy[flatten(x, y, z)] = 1;
    }

    for (int x = 0; x < grid_size_.x(); ++x) {
        for (int y = 0; y < grid_size_.y(); ++y) {
            for (int z = 0; z < grid_size_.z(); ++z) {
                const float voxel_center_z = min_bound_.z() + (static_cast<float>(z) + 0.5f) * resolution_;
                if (voxel_center_z <= 0.0f) {
                    occupancy[flatten(x, y, z)] = 1;
                }
            }
        }
    }
}

void EsdfMap::buildSignedDistanceField(const std::vector<uint8_t> &occupancy)
{
    const size_t total_size = occupancy.size();
    std::vector<float> outside(total_size, kLargeDistance);
    std::vector<float> inside(total_size, kLargeDistance);

    for (size_t i = 0; i < total_size; ++i) {
        if (occupancy[i]) {
            outside[i] = 0.0f;
        } else {
            inside[i] = 0.0f;
        }
    }

    distanceTransform3D(outside);
    distanceTransform3D(inside);

    sdf_.resize(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        const float outside_dist = std::sqrt(outside[i]) * resolution_;
        const float inside_dist = std::sqrt(inside[i]) * resolution_;
        sdf_[i] = occupancy[i] ? -inside_dist : outside_dist;
    }
}

void EsdfMap::distanceTransform3D(std::vector<float> &grid) const
{
    std::vector<float> input_line;
    std::vector<float> output_line;

    input_line.reserve(std::max({grid_size_.x(), grid_size_.y(), grid_size_.z()}));
    output_line.reserve(std::max({grid_size_.x(), grid_size_.y(), grid_size_.z()}));

    for (int y = 0; y < grid_size_.y(); ++y) {
        for (int z = 0; z < grid_size_.z(); ++z) {
            input_line.assign(grid_size_.x(), 0.0f);
            output_line.assign(grid_size_.x(), 0.0f);
            for (int x = 0; x < grid_size_.x(); ++x) {
                input_line[x] = grid[flatten(x, y, z)];
            }
            distanceTransform1D(input_line, output_line);
            for (int x = 0; x < grid_size_.x(); ++x) {
                grid[flatten(x, y, z)] = output_line[x];
            }
        }
    }

    for (int x = 0; x < grid_size_.x(); ++x) {
        for (int z = 0; z < grid_size_.z(); ++z) {
            input_line.assign(grid_size_.y(), 0.0f);
            output_line.assign(grid_size_.y(), 0.0f);
            for (int y = 0; y < grid_size_.y(); ++y) {
                input_line[y] = grid[flatten(x, y, z)];
            }
            distanceTransform1D(input_line, output_line);
            for (int y = 0; y < grid_size_.y(); ++y) {
                grid[flatten(x, y, z)] = output_line[y];
            }
        }
    }

    for (int x = 0; x < grid_size_.x(); ++x) {
        for (int y = 0; y < grid_size_.y(); ++y) {
            input_line.assign(grid_size_.z(), 0.0f);
            output_line.assign(grid_size_.z(), 0.0f);
            for (int z = 0; z < grid_size_.z(); ++z) {
                input_line[z] = grid[flatten(x, y, z)];
            }
            distanceTransform1D(input_line, output_line);
            for (int z = 0; z < grid_size_.z(); ++z) {
                grid[flatten(x, y, z)] = output_line[z];
            }
        }
    }
}

void EsdfMap::distanceTransform1D(const std::vector<float> &input, std::vector<float> &output)
{
    const int n = static_cast<int>(input.size());
    std::vector<int> locations(n, 0);
    std::vector<float> boundaries(n + 1, 0.0f);

    int k = 0;
    locations[0] = 0;
    boundaries[0] = -kLargeDistance;
    boundaries[1] = kLargeDistance;

    for (int q = 1; q < n; ++q) {
        float intersection = 0.0f;
        while (true) {
            const int previous = locations[k];
            intersection = ((input[q] + static_cast<float>(q * q)) -
                            (input[previous] + static_cast<float>(previous * previous))) /
                           (2.0f * static_cast<float>(q - previous));
            if (intersection > boundaries[k] || k == 0) {
                break;
            }
            --k;
        }
        ++k;
        locations[k] = q;
        boundaries[k] = intersection;
        boundaries[k + 1] = kLargeDistance;
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (boundaries[k + 1] < static_cast<float>(q)) {
            ++k;
        }
        const float diff = static_cast<float>(q - locations[k]);
        output[q] = diff * diff + input[locations[k]];
    }
}
