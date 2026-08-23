// PCL (C++) baseline harness for the pcl-rustic comparison benchmarks.
//
// Reads the exact point data the Python benchmarks use, so the comparison is
// not confounded by different inputs. Input format is trivial:
//
//   int64 n, then n * 3 little-endian float64 (x, y, z interleaved)
//
// Usage: pcl_bench <data-file> <leaf-size>
//
// Timings cover the operation only; building the pcl::PointCloud from the
// file happens outside every timed region, matching how the Python side
// times Open3D and pcl-rustic.
//
// Note on types: PCL's PointXYZ stores f32 (padded to 16 bytes), Open3D
// stores f64, and pcl-rustic stores f32 relative to an f64 offset. PCL is
// therefore doing the least precise arithmetic of the three.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

using Clock = std::chrono::steady_clock;

static double seconds_since(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <data-file> <leaf-size>\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  const float leaf = std::strtof(argv[2], nullptr);

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  std::int64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  std::vector<double> raw(static_cast<std::size_t>(n) * 3);
  in.read(reinterpret_cast<char*>(raw.data()),
          static_cast<std::streamsize>(raw.size() * sizeof(double)));
  if (!in) {
    std::fprintf(stderr, "short read from %s\n", path);
    return 2;
  }

  // Cloud construction happens outside the timers.
  auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->resize(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    (*cloud)[i].x = static_cast<float>(raw[i * 3 + 0]);
    (*cloud)[i].y = static_cast<float>(raw[i * 3 + 1]);
    (*cloud)[i].z = static_cast<float>(raw[i * 3 + 2]);
  }

  // --- voxel grid -----------------------------------------------------
  pcl::PointCloud<pcl::PointXYZ> down;
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  grid.setInputCloud(cloud);
  grid.setLeafSize(leaf, leaf, leaf);
  auto t0 = Clock::now();
  grid.filter(down);
  const double voxel_secs = seconds_since(t0);

  // PCL's VoxelGrid indexes voxels with a 32-bit integer and silently
  // returns the input untouched when the grid would overflow it. Report the
  // output size so that case is visible instead of masquerading as a fast run.
  const bool voxel_refused = down.size() == cloud->size();

  // --- transform ------------------------------------------------------
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m(0, 0) = 0.8f;  m(0, 1) = -0.6f;
  m(1, 0) = 0.6f;  m(1, 1) = 0.8f;
  m(0, 3) = 1.0f;  m(1, 3) = 2.0f;  m(2, 3) = 3.0f;

  pcl::PointCloud<pcl::PointXYZ> transformed;
  t0 = Clock::now();
  pcl::transformPointCloud(*cloud, transformed, m);
  const double transform_secs = seconds_since(t0);

  std::printf("n=%lld voxel_secs=%.6f voxel_out=%zu voxel_refused=%d transform_secs=%.6f\n",
              static_cast<long long>(n), voxel_secs, down.size(),
              voxel_refused ? 1 : 0, transform_secs);
  return 0;
}
