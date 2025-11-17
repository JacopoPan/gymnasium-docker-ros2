import zmq
import threading
import time
import struct
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped

import gz.transport13
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean as GzBoolean


ZMQ_PORT = 5555

class ZMQBridge(Node):
    def __init__(self):
        super().__init__('zmq_bridge_node')
        
        # ZeroMQ Setup 
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP) 
        self.socket.bind(f"tcp://0.0.0.0:{ZMQ_PORT}")
        self.get_logger().info(f"ZMQ REP socket bound to port {ZMQ_PORT}")
        
        # ROS 2 Setup
        self.action_publisher = self.create_publisher(Float64, '/control_input', 10)
        self.state_subscriber = self.create_subscription(
            Vector3Stamped, 
            '/state', 
            self.state_callback, 
            10 # Use a QoS of 10
        )
        self.latest_state = None
        self.state_lock = threading.Lock() # Use a threading.Lock for safe access to self.latest_state
        self.state_event = threading.Event()

        # GZ Transport Setup
        self.gz_node = gz.transport13.Node()
        
        # Start thread to handle ZMQ requests
        self.zmq_thread = threading.Thread(target=self.zmq_listener, daemon=True)
        self.zmq_thread.start()
        self.get_logger().info("ZMQ listener thread started.")

    def state_callback(self, msg):
        with self.state_lock:
            self.latest_state = (
                msg.vector.x,
                msg.vector.y,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec
            )
        self.state_event.set() # SIGNAL the zmq_listener thread

    def zmq_listener(self):        
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        while rclpy.ok():
            socks = dict(poller.poll(None)) # Wait for a request
            
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                try:
                    # Receive the action (REQ)
                    action_bytes = self.socket.recv(zmq.NOBLOCK)
                    force = struct.unpack('d', action_bytes)[0]
                    # self.get_logger().info(f"Received action: {force:.4f}")

                    # Clear old state and publish new action
                    with self.state_lock:
                        self.latest_state = None
                    action_msg = Float64(data=force)
                    self.action_publisher.publish(action_msg)

                    # Call Gazebo service to step the simulation
                    req = WorldControl()
                    req.multi_step = 1 # 50ms (see gdr2-world.sdf) = 20Hz (see dynamics_node.py)
                    req.pause = True
                    result, response = self.gz_node.request(
                        "/world/gdr2-world/control",
                        req,
                        WorldControl,
                        GzBoolean,
                        1000 # 1 second timeout
                    )

                    # Wait for the new state to arrive from the /state topic
                    self.state_event.clear()
                    new_state_received = self.state_event.wait(timeout=2.0)
                    # Return the state (REP)
                    if new_state_received:
                        pos, vel, sec, nanosec = self.latest_state
                        reply_payload = struct.pack('ddii', pos, vel, sec, nanosec)
                    else:
                        self.get_logger().warn("State timeout!")
                        # reply_payload = b"0.0,0.0,0,0" # Fail safe
                    
                    self.socket.send(reply_payload)

                except zmq.error.Again:
                    pass
                except Exception as e:
                    self.get_logger().error(f"Bridge critical error: {e}")
                    self.socket.send(b"ERROR") 

def main(args=None):
    rclpy.init(args=args)
    node = ZMQBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
