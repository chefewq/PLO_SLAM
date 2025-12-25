#include "MapObject.h"
#include <Eigen/Dense>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <mutex>
#include <opencv2/core/base.hpp>
#include <vector>

namespace ORB_SLAM3 {
MapObject::MapObject() {}
MapObject::MapObject(const int trackId, const int plabel, Ellipsoid &pellipsoid, Map *map)
    : track_id(trackId),  mpMap(map),label(plabel) {
      mpEllipsoid = pellipsoid;
    }

Map *MapObject::GetMap() {
  unique_lock<mutex> lock(mMutexMap);
  return mpMap;
}

void MapObject::SetEllipsoid(Ellipsoid ell){
  std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
  mpEllipsoid = ell;
  kf_x = Eigen::VectorXd::Zero(6);
  kf_P = Eigen::MatrixXd::Identity(6,6) * 0.01;
}

Ellipsoid &MapObject::GetEllipsoid() {
  // std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
  return mpEllipsoid;
}



template<class Archive>
void MapObject::serialize(Archive & ar, const unsigned int version){

    UNUSED_VAR(version);
    using namespace boost::serialization; 
    ar & track_id;
    ar & mpMap; // Luigi: added this 
    
}


template void MapObject::serialize(boost::archive::binary_iarchive &,
                                   const unsigned int);
template void MapObject::serialize(boost::archive::binary_oarchive &,
                                   const unsigned int);
template void MapObject::serialize(boost::archive::text_iarchive &,
                                   const unsigned int);
template void MapObject::serialize(boost::archive::text_oarchive &,
                                   const unsigned int);

} // namespace ORB_SLAM3