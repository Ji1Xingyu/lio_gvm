#include "fast_vgicp/lsq_registration.hpp"
#include "fast_vgicp/impl/lsq_registration_impl.hpp"

template class fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::LsqRegistration<pcl::PointXYZI, pcl::PointXYZI>;
template class fast_gicp::LsqRegistration<pcl::PointXYZINormal, pcl::PointXYZINormal>;
