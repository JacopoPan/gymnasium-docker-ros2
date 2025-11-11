import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped 
import zmq
import threading
import time

ZMQ_PORT = 5555

class ZMQBridge(Node):
    def __init__(self):
        super().__init__('zmq_bridge_node')
        
        # ZeroMQ Setup (PULL Socket)
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PULL) 
        self.socket.bind(f"tcp://0.0.0.0:{ZMQ_PORT}")
        self.get_logger().info(f"ZMQ PULL socket bound to port {ZMQ_PORT}")
        
        self.action_publisher = self.create_publisher(Float64, '/control_input', 10)
        
        # Start thread to handle ZMQ requests
        self.zmq_thread = threading.Thread(target=self.zmq_listener, daemon=True)
        self.zmq_thread.start()

    def zmq_listener(self):
        # PULL sockets are blocking, but that's fine for a dedicated thread.
        while rclpy.ok():
            try:
                # 1. Wait for the action (PULL) from the host
                action_bytes = self.socket.recv() # This call blocks until a message arrives.
                
                force = float(action_bytes.decode('utf-8'))
                self.get_logger().info(f"Received action: {force:.4f}. Publishing to ROS.")
                
                # 2. Publish the action to the ROS network
                action_msg = Float64(data=force)
                self.action_publisher.publish(action_msg)

            except zmq.error.ZMQError as e:
                self.get_logger().error(f"ZMQ Error: {e}")
                break
            except Exception as e:
                self.get_logger().error(f"Bridge critical error: {e}")
                
def main(args=None):
    rclpy.init(args=args)
    node = ZMQBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
