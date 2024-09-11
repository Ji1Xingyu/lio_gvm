


#include "fast_vgicp/fast_vgicp.hpp"
#include "fast_vgicp/impl/fast_vgicp_impl.hpp"

template class fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI>;
template class fast_gicp::FastVGICP<pcl::PointXYZINormal, pcl::PointXYZINormal>;
