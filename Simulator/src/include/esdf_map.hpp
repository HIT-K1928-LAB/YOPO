#ifndef ESDF_MAP_HPP
#define ESDF_MAP_HPP

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

struct EsdfMapConfig {
    float resolution{0.2f};
    Eigen::Vector3f expand_min{0.0f, 0.0f, 0.2f};
    Eigen::Vector3f expand_max{0.0f, 0.0f, 6.0f};
};

class EsdfMap {
public:
    EsdfMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const EsdfMapConfig &config);

    float queryDistance(const Eigen::Vector3f &pos) const;

    bool isCollision(const Eigen::Vector3f &pos, float radius) const {
        return queryDistance(pos) <= radius;
    }

    const Eigen::Vector3f &minBound() const { return min_bound_; }
    const Eigen::Vector3f &maxBound() const { return max_bound_; }
    const Eigen::Vector3i &gridSize() const { return grid_size_; }
    float resolution() const { return resolution_; }

private:
    int flatten(int x, int y, int z) const;
    float mirroredCoordinate(float coord, int length) const;
    float sampleVoxel(int x, int y, int z) const;
    float trilinearSample(float gx, float gy, float gz) const;

    void buildOccupancy(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<uint8_t> &occupancy) const;
    void buildSignedDistanceField(const std::vector<uint8_t> &occupancy);
    void distanceTransform3D(std::vector<float> &grid) const;
    static void distanceTransform1D(const std::vector<float> &input, std::vector<float> &output);

    float resolution_{0.2f};
    float max_clearance_{0.0f};
    Eigen::Vector3f min_bound_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f max_bound_{Eigen::Vector3f::Zero()};
    Eigen::Vector3i grid_size_{Eigen::Vector3i::Zero()};
    std::vector<float> sdf_;
};

#endif
