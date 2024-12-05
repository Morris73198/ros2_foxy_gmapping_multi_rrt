#include "slam_gmapping/slam_gmapping.h"
#include "tf2_ros/create_timer_ros.h"

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

using std::placeholders::_1;

SlamGmapping::SlamGmapping():
    Node("slam_gmapping"),
    scan_filter_sub_(nullptr),
    scan_filter_(nullptr),
    laser_count_(0),
    transform_thread_(nullptr)
{
    buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        get_node_base_interface(),
        get_node_timers_interface());
    buffer_->setCreateTimerInterface(timer_interface);
    tfl_ = std::make_shared<tf2_ros::TransformListener>(*buffer_);
    node_ = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
    tfB_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    map_to_odom_.setIdentity();
    seed_ = static_cast<unsigned long>(time(nullptr));
    init();
    startLiveSlam();
}

void SlamGmapping::init()
{
    throttle_scans_ = 1;
    base_frame_ = "base_link";
    map_frame_ = "map";
    odom_frame_ = "odom";
    transform_publish_period_ = 0.05;

    this->declare_parameter("clear_radius", 4);
    clear_radius_ = this->get_parameter("clear_radius").as_int();

    robot1_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/robot1_pose", 10,
        std::bind(&SlamGmapping::robot1PoseCallback, this, _1));
    
    robot2_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/robot2_pose", 10,
        std::bind(&SlamGmapping::robot2PoseCallback, this, _1));

    gsp_ = new GMapping::GridSlamProcessor();
    gsp_laser_ = nullptr;
    gsp_odom_ = nullptr;
    got_first_scan_ = false;
    got_map_ = false;
    
    map_update_interval_ = tf2::durationFromSec(0.5);
    maxUrange_ = 80.0;  
    maxRange_ = 0.0;
    minimum_score_ = 0;
    sigma_ = 0.05;
    kernelSize_ = 1;
    lstep_ = 0.05;
    astep_ = 0.05;
    iterations_ = 5;
    lsigma_ = 0.075;
    ogain_ = 3.0;
    lskip_ = 0;
    srr_ = 0.1;
    srt_ = 0.2;
    str_ = 0.1;
    stt_ = 0.2;
    linearUpdate_ = 1.0;
    angularUpdate_ = 0.5;
    temporalUpdate_ = 1.0;
    resampleThreshold_ = 0.5;
    particles_ = 30;
    xmin_ = -10.0;
    ymin_ = -10.0;
    xmax_ = 10.0;
    ymax_ = 10.0;
    delta_ = 0.05;
    occ_thresh_ = 0.25;
    llsamplerange_ = 0.01;
    llsamplestep_ = 0.01;
    lasamplerange_ = 0.005;
    lasamplestep_ = 0.005;
    tf_delay_ = transform_publish_period_;
}

bool SlamGmapping::getOdomPose(GMapping::OrientedPoint& gmap_pose, const rclcpp::Time& t)
{
    centered_laser_pose_.header.stamp = t;
    geometry_msgs::msg::PoseStamped odom_pose;
    try
    {
        buffer_->transform(centered_laser_pose_, odom_pose, odom_frame_, tf2::durationFromSec(1.0));
    }
    catch(tf2::TransformException& e)
    {
        RCLCPP_WARN(this->get_logger(), "Failed to compute odom pose, skipping scan (%s)", e.what());
        return false;
    }

    double yaw = tf2::getYaw(odom_pose.pose.orientation);
    gmap_pose = GMapping::OrientedPoint(odom_pose.pose.position.x,
                                      odom_pose.pose.position.y,
                                      yaw);
    return true;
}

bool SlamGmapping::initMapper(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
    laser_frame_ = scan->header.frame_id;
    geometry_msgs::msg::PoseStamped ident;
    geometry_msgs::msg::PoseStamped laser_pose;
    
    try {
        ident.header.frame_id = laser_frame_;
        ident.header.stamp = scan->header.stamp;
        tf2::Transform transform;
        transform.setIdentity();
        tf2::toMsg(transform, ident.pose);
        buffer_->transform(ident, laser_pose, base_frame_);
    }
    catch(tf2::TransformException& e) {
        RCLCPP_WARN(this->get_logger(), "Failed to compute laser pose, aborting initialization (%s)", e.what());
        return false;
    }

    geometry_msgs::msg::PointStamped up;
    up.header.stamp = scan->header.stamp;
    up.header.frame_id = base_frame_;
    up.point.x = up.point.y = 0;
    up.point.z = 1 + laser_pose.pose.position.z;
    
    try {
        buffer_->transform(up, up, laser_frame_);
    }
    catch(tf2::TransformException& e) {
        RCLCPP_WARN(this->get_logger(), "Unable to determine orientation of laser: %s", e.what());
        return false;
    }

    if (fabs(fabs(up.point.z) - 1) > 0.001) {
        RCLCPP_INFO(this->get_logger(), "Laser has to be mounted planar! Z-coordinate has to be 1 or -1, but gave: %.5f", up.point.z);
        return false;
    }

    gsp_laser_beam_count_ = static_cast<unsigned int>(scan->ranges.size());
    double angle_center = (scan->angle_min + scan->angle_max)/2;

    centered_laser_pose_.header.frame_id = laser_frame_;
    centered_laser_pose_.header.stamp = get_clock()->now();
    tf2::Quaternion q;

    if (up.point.z > 0) {
        do_reverse_range_ = scan->angle_min > scan->angle_max;
        q.setEuler(angle_center, 0, 0);
        RCLCPP_INFO(this->get_logger(), "Laser is mounted upwards.");
    } else {
        do_reverse_range_ = scan->angle_min < scan->angle_max;
        q.setEuler(-angle_center, 0, M_PI);
        RCLCPP_INFO(this->get_logger(), "Laser is mounted upside down.");
    }

    centered_laser_pose_.pose.orientation = tf2::toMsg(q);
    centered_laser_pose_.pose.position.x = 0;
    centered_laser_pose_.pose.position.y = 0;
    centered_laser_pose_.pose.position.z = 0;

    laser_angles_.resize(scan->ranges.size());
    double theta = - std::fabs(scan->angle_min - scan->angle_max)/2;
    for(unsigned int i=0; i < scan->ranges.size(); ++i) {
        laser_angles_[i] = theta;
        theta += std::fabs(scan->angle_increment);
    }

    GMapping::OrientedPoint gmap_pose(0, 0, 0);

    maxRange_ = scan->range_max - 0.01;
    maxUrange_ = maxRange_;

    gsp_laser_ = new GMapping::RangeSensor("FLASER",
                                         gsp_laser_beam_count_,
                                         fabs(scan->angle_increment),
                                         gmap_pose,
                                         0.0,
                                         maxRange_);

    GMapping::SensorMap smap;
    smap.insert(make_pair(gsp_laser_->getName(), gsp_laser_));
    gsp_->setSensorMap(smap);

    gsp_odom_ = new GMapping::OdometrySensor(odom_frame_);

    GMapping::OrientedPoint initialPose;
    if(!getOdomPose(initialPose, scan->header.stamp))
    {
        RCLCPP_WARN(this->get_logger(), "Unable to determine initial pose! Starting point will be set to zero.");
        initialPose = GMapping::OrientedPoint(0.0, 0.0, 0.0);
    }

    gsp_->setMatchingParameters(maxUrange_, maxRange_, sigma_,
                             kernelSize_, lstep_, astep_, iterations_,
                             lsigma_, ogain_, lskip_);

    gsp_->setMotionModelParameters(srr_, srt_, str_, stt_);
    gsp_->setUpdateDistances(linearUpdate_, angularUpdate_, resampleThreshold_);
    gsp_->setUpdatePeriod(temporalUpdate_);
    gsp_->setgenerateMap(false);
    gsp_->GridSlamProcessor::init(particles_, xmin_, ymin_, xmax_, ymax_,
                               delta_, initialPose);
    gsp_->setllsamplerange(llsamplerange_);
    gsp_->setllsamplestep(llsamplestep_);
    gsp_->setlasamplerange(lasamplerange_);
    gsp_->setlasamplestep(lasamplestep_);
    gsp_->setminimumScore(minimum_score_);

    GMapping::sampleGaussian(1, seed_);

    RCLCPP_INFO(this->get_logger(), "Initialization complete");

    return true;
}

bool SlamGmapping::addScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan, GMapping::OrientedPoint& gmap_pose)
{
    if(!getOdomPose(gmap_pose, scan->header.stamp))
        return false;

    if(scan->ranges.size() != gsp_laser_beam_count_)
        return false;

    double* ranges_double = new double[scan->ranges.size()];
    if(do_reverse_range_) {
        for(int i = 0; i < static_cast<int>(scan->ranges.size()); i++) {
            if(scan->ranges[scan->ranges.size() - i - 1] < scan->range_min)
                ranges_double[i] = scan->range_max;
            else
                ranges_double[i] = scan->ranges[scan->ranges.size() - i - 1];
        }
    } else {
        for(unsigned int i = 0; i < scan->ranges.size(); i++) {
            if(scan->ranges[i] < scan->range_min)
                ranges_double[i] = scan->range_max;
            else
                ranges_double[i] = scan->ranges[i];
        }
    }

    GMapping::RangeReading reading(static_cast<unsigned int>(scan->ranges.size()),
                                 ranges_double,
                                 gsp_laser_,
                                 scan->header.stamp.sec);
    delete[] ranges_double;
    reading.setPose(gmap_pose);
    return gsp_->processScan(reading);
}

void SlamGmapping::robot1PoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    RCLCPP_INFO(get_logger(), "Received robot1 pose: x=%.2f, y=%.2f", 
               msg->pose.position.x, msg->pose.position.y);
    std::lock_guard<std::mutex> lock(pose_mutex_);
    robot1_history_.push_back(*msg);
    if (robot1_history_.size() > MAX_HISTORY_SIZE) {
        robot1_history_.pop_front();
    }
}

void SlamGmapping::robot2PoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    RCLCPP_INFO(get_logger(), "Received robot2 pose: x=%.2f, y=%.2f", 
               msg->pose.position.x, msg->pose.position.y);
    std::lock_guard<std::mutex> lock(pose_mutex_);
    robot2_history_.push_back(*msg);
    if (robot2_history_.size() > MAX_HISTORY_SIZE) {
        robot2_history_.pop_front();
    }
}

void SlamGmapping::clearRobotTraces(nav_msgs::msg::OccupancyGrid& map)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    
    RCLCPP_INFO(get_logger(), "Map info: resolution=%.3f, origin=(%.2f,%.2f), size=%dx%d",
                map.info.resolution, 
                map.info.origin.position.x,
                map.info.origin.position.y,
                map.info.width,
                map.info.height);

    auto clearPose = [&](const geometry_msgs::msg::PoseStamped& pose_stamped) {
        const auto& pose = pose_stamped.pose;
        double map_x = pose.position.x - map.info.origin.position.x;
        double map_y = pose.position.y - map.info.origin.position.y;
        
        int x = static_cast<int>(map_x / map.info.resolution);
        int y = static_cast<int>(map_y / map.info.resolution);
        
        RCLCPP_INFO(get_logger(), "Clearing around point (%.2f,%.2f) -> grid(%d,%d)", 
                    pose.position.x, pose.position.y, x, y);

        int cells_cleared = 0;
        for (int dx = -clear_radius_; dx <= clear_radius_; dx++) {
            for (int dy = -clear_radius_; dy <= clear_radius_; dy++) {
                if (dx*dx + dy*dy <= clear_radius_*clear_radius_) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < static_cast<int>(map.info.width) &&
                        ny >= 0 && ny < static_cast<int>(map.info.height)) {
                        int idx = ny * map.info.width + nx;
                        // 清除任何非未知的占用格
                        if (map.data[idx] > 50) {
                            map.data[idx] = 0;
                            cells_cleared++;
                        }
                    }
                }
            }
        }
        if (cells_cleared > 0) {
            RCLCPP_INFO(get_logger(), "Cleared %d cells", cells_cleared);
        }
    };

    for (const auto& pose : robot1_history_) {
        clearPose(pose);
    }
    for (const auto& pose : robot2_history_) {
        clearPose(pose);
    }
}

void SlamGmapping::startLiveSlam()
{
    entropy_publisher_ = this->create_publisher<std_msgs::msg::Float64>("entropy", rclcpp::SystemDefaultsQoS());
    sst_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", 
        rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());
    sstm_ = this->create_publisher<nav_msgs::msg::MapMetaData>("map_metadata", 
        rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

    scan_filter_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>
        (node_, "scan", rclcpp::SensorDataQoS().get_rmw_qos_profile());
    scan_filter_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>
        (*scan_filter_sub_, *buffer_, odom_frame_, 10, node_);
    scan_filter_->registerCallback(std::bind(&SlamGmapping::laserCallback, this, _1));

    transform_thread_ = std::make_shared<std::thread>
        (std::bind(&SlamGmapping::publishLoop, this, transform_publish_period_));
}

void SlamGmapping::publishLoop(double transform_publish_period)
{
    if (transform_publish_period == 0)
        return;
    rclcpp::Rate r(1.0 / transform_publish_period);
    while (rclcpp::ok()) {
        publishTransform();
        r.sleep();
    }
}

void SlamGmapping::publishTransform()
{
    map_to_odom_mutex_.lock();
    rclcpp::Time tf_expiration = get_clock()->now() + rclcpp::Duration(
            static_cast<int32_t>(static_cast<rcl_duration_value_t>(tf_delay_)), 0);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = map_frame_;
    transform.header.stamp = tf_expiration;
    transform.child_frame_id = odom_frame_;
    
    transform.transform = tf2::toMsg(map_to_odom_);
    tfB_->sendTransform(transform);
    map_to_odom_mutex_.unlock();
}

void SlamGmapping::updateMap(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
    map_mutex_.lock();
    GMapping::ScanMatcher matcher;
    
    matcher.setLaserParameters(static_cast<unsigned int>(scan->ranges.size()),
                             &(laser_angles_[0]),
                             gsp_laser_->getPose());

    matcher.setlaserMaxRange(maxRange_);
    matcher.setusableRange(maxUrange_);
    matcher.setgenerateMap(true);

    GMapping::GridSlamProcessor::Particle best =
            gsp_->getParticles()[gsp_->getBestParticleIndex()];
            
    std_msgs::msg::Float64 entropy;
    entropy.data = computePoseEntropy();
    if(entropy.data > 0.0)
        entropy_publisher_->publish(entropy);

    if(!got_map_) {
        map_.info.resolution = static_cast<nav_msgs::msg::MapMetaData::_resolution_type>(delta_);
        map_.info.origin.position.x = 0.0;
        map_.info.origin.position.y = 0.0;
        map_.info.origin.position.z = 0.0;
        map_.info.origin.orientation.x = 0.0;
        map_.info.origin.orientation.y = 0.0;
        map_.info.origin.orientation.z = 0.0;
        map_.info.origin.orientation.w = 1.0;
    }

    GMapping::Point center;
    center.x=(xmin_ + xmax_) / 2.0;
    center.y=(ymin_ + ymax_) / 2.0;

    GMapping::ScanMatcherMap smap(center, xmin_, ymin_, xmax_, ymax_, delta_);

    for(GMapping::GridSlamProcessor::TNode* n = best.node; n; n = n->parent) {
        if(!n->reading) {
            continue;
        }
        matcher.invalidateActiveArea();
        matcher.computeActiveArea(smap, n->pose, &((*n->reading)[0]));
        matcher.registerScan(smap, n->pose, &((*n->reading)[0]));
    }

    if(map_.info.width != (unsigned int) smap.getMapSizeX() || 
       map_.info.height != (unsigned int) smap.getMapSizeY()) {

        GMapping::Point wmin = smap.map2world(GMapping::IntPoint(0, 0));
        GMapping::Point wmax = smap.map2world(GMapping::IntPoint(smap.getMapSizeX(), smap.getMapSizeY()));
        xmin_ = wmin.x; ymin_ = wmin.y;
        xmax_ = wmax.x; ymax_ = wmax.y;

        map_.info.width = static_cast<nav_msgs::msg::MapMetaData::_width_type>(smap.getMapSizeX());
        map_.info.height = static_cast<nav_msgs::msg::MapMetaData::_height_type>(smap.getMapSizeY());
        map_.info.origin.position.x = xmin_;
        map_.info.origin.position.y = ymin_;
        map_.data.resize(map_.info.width * map_.info.height);
    }

    for(int x=0; x < smap.getMapSizeX(); x++) {
        for(int y=0; y < smap.getMapSizeY(); y++) {
            GMapping::IntPoint p(x, y);
            double occ=smap.cell(p);
            assert(occ <= 1.0);
            if(occ < 0)
                map_.data[MAP_IDX(map_.info.width, x, y)] = -1;
            else if(occ > occ_thresh_)
                map_.data[MAP_IDX(map_.info.width, x, y)] = 100;
            else
                map_.data[MAP_IDX(map_.info.width, x, y)] = 0;
        }
    }

    clearRobotTraces(map_);

    got_map_ = true;
    map_.header.stamp = get_clock()->now();
    map_.header.frame_id = map_frame_;
    
    sst_->publish(map_);
    sstm_->publish(map_.info);
    map_mutex_.unlock();
}

void SlamGmapping::laserCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
    laser_count_++;
    if ((laser_count_ % throttle_scans_) != 0)
        return;

    static tf2::TimePoint last_map_update = tf2::TimePointZero;

    if(!got_first_scan_) {
        if(!initMapper(scan))
            return;
        got_first_scan_ = true;
    }

    GMapping::OrientedPoint odom_pose;

    if(addScan(scan, odom_pose)) {
        GMapping::OrientedPoint mpose = gsp_->getParticles()[gsp_->getBestParticleIndex()].pose;

        tf2::Quaternion q;
        q.setRPY(0, 0, mpose.theta);
        tf2::Transform laser_to_map = tf2::Transform(q, tf2::Vector3(mpose.x, mpose.y, 0.0)).inverse();
        q.setRPY(0, 0, odom_pose.theta);
        tf2::Transform odom_to_laser = tf2::Transform(q, tf2::Vector3(odom_pose.x, odom_pose.y, 0.0));

        map_to_odom_mutex_.lock();
        map_to_odom_ = (odom_to_laser * laser_to_map).inverse();
        map_to_odom_mutex_.unlock();

        tf2::TimePoint timestamp = tf2_ros::fromMsg(scan->header.stamp);
        if(!got_map_ || (timestamp - last_map_update) > map_update_interval_) {
            updateMap(scan);
            last_map_update = tf2_ros::fromMsg(scan->header.stamp);
        }
    }
}

double SlamGmapping::computePoseEntropy()
{
    double weight_total = 0.0;
    for(const auto& it : gsp_->getParticles()) {
        weight_total += it.weight;
    }
    double entropy = 0.0;
    for(const auto& it : gsp_->getParticles()) {
        if(it.weight/weight_total > 0.0)
            entropy += it.weight/weight_total * log(it.weight/weight_total);
    }
    return -entropy;
}

SlamGmapping::~SlamGmapping()
{
    if(transform_thread_){
        transform_thread_->join();
    }

    delete gsp_;
    if(gsp_laser_)
        delete gsp_laser_;
    if(gsp_odom_)
        delete gsp_odom_;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto slam_gmapping = std::make_shared<SlamGmapping>();
    rclcpp::spin(slam_gmapping);
    rclcpp::shutdown();
    return 0;
}