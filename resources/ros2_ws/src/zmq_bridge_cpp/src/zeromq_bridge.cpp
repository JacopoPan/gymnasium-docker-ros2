#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstring>
// ROS 2 Includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
// Gazebo Includes (libgz-transport13-*, libgz-msgs10-dev)
#include <gz/transport/Node.hh>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/boolean.pb.h>
// ZeroMQ Includes (requires cppzmq installed with libzmq3-dev)
#include <zmq.hpp> 

using namespace std::chrono_literals;

// Structure matches Python struct.pack('ddii')
// d = double (8 bytes), i = int (4 bytes)
// Use #pragma pack to ensure no padding is added by compiler
#pragma pack(push, 1)
struct StatePayload {
    double x;
    double y;
    int32_t sec;
    int32_t nanosec;
};
#pragma pack(pop)

class ZMQBridge : public rclcpp::Node {
public:
    ZMQBridge() : Node("zmq_bridge_node"), context_(1), socket_(context_, zmq::socket_type::rep) {
        
        // 1. ZMQ Setup
        socket_.bind("tcp://*:5555"); // '*' binds to all interfaces (equivalent to 0.0.0.0)
        RCLCPP_INFO(this->get_logger(), "ZMQ REP socket bound to port 5555");

        // 2. ROS 2 Setup
        publisher_ = this->create_publisher<std_msgs::msg::Float64>("/control_input", 10);
        
        subscription_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/state", 10, 
            std::bind(&ZMQBridge::state_callback, this, std::placeholders::_1));

        // 3. Start ZMQ Thread
        running_ = true;
        zmq_thread_ = std::thread(&ZMQBridge::zmq_listener, this);
    }

    ~ZMQBridge() {
        running_ = false;
        // Clean shutdown for ZMQ
        context_.close(); 
        if (zmq_thread_.joinable()) {
            zmq_thread_.join();
        }
    }

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr subscription_;
    
    gz::transport::Node gz_node_;

    std::thread zmq_thread_;
    std::atomic<bool> running_;

    std::mutex state_mutex_;
    std::condition_variable state_cv_;
    bool state_ready_ = false;
    StatePayload current_state_;

    void state_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        current_state_.x = msg->vector.x;
        current_state_.y = msg->vector.y;
        current_state_.sec = msg->header.stamp.sec;
        current_state_.nanosec = msg->header.stamp.nanosec;
        
        state_ready_ = true;
        state_cv_.notify_one(); // Signal the waiting ZMQ thread
    }

    void zmq_listener() {
        RCLCPP_INFO(this->get_logger(), "ZMQ listener thread started.");
        
        zmq::pollitem_t items[] = {
            { static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0 }
        };

        while (rclcpp::ok() && running_) {
            zmq::poll(&items[0], 1, 100); // Poll with 100ms timeout to allow checking running_ flag

            if (items[0].revents & ZMQ_POLLIN) {
                try {
                    // 1. Receive Action
                    zmq::message_t request;
                    auto res = socket_.recv(request, zmq::recv_flags::none);
                    if (!res) continue;
                    double force = *static_cast<double*>(request.data()); // Get force (double)

                    // 2. Reset state flag and Publish Action
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_ready_ = false;
                    }
                    auto ros_msg = std_msgs::msg::Float64();
                    ros_msg.data = force;
                    publisher_->publish(ros_msg);

                    // 3. Step Gazebo
                    gz::msgs::WorldControl req;
                    req.set_multi_step(1); // 50ms (see gdr2-world.sdf) = 20Hz (see dynamics_node.py)
                    req.set_pause(true);
                    gz::msgs::Boolean rep;
                    bool result;
                    unsigned int timeout = 1000; 
                    bool executed = gz_node_.Request("/world/gdr2-world/control", req, timeout, rep, result);
                    if (!executed) {
                        RCLCPP_WARN(this->get_logger(), "Gazebo service call failed or timed out.");
                    }

                    // 4. Wait for ROS State
                    std::unique_lock<std::mutex> lock(state_mutex_);
                    bool received = state_cv_.wait_for(lock, 2000ms, [this]{ return state_ready_; }); // Wait up to 2 seconds for state_ready_ to become true

                    // 5. Send Reply
                    zmq::message_t reply(sizeof(StatePayload));
                    if (received) {
                        // Copy struct directly into ZMQ message buffer
                        std::memcpy(reply.data(), &current_state_, sizeof(StatePayload));
                    } else {
                        RCLCPP_WARN(this->get_logger(), "State timeout!");
                        // Send zeros or error code
                        StatePayload empty = {0.0, 0.0, 0, 0};
                        std::memcpy(reply.data(), &empty, sizeof(StatePayload));
                    }
                    socket_.send(reply, zmq::send_flags::none);

                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "ZMQ Error: %s", e.what());
                }
            }
        }
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZMQBridge>());
    rclcpp::shutdown();
    return 0;
}
