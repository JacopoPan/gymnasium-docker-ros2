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

    def zmq_listener(self):        
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        while rclpy.ok():
            socks = dict(poller.poll(100)) # Wait for a request with a 100ms timeout
            
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                try:
                    # Receive the action (REQ)
                    action_bytes = self.socket.recv(zmq.NOBLOCK)
                    force = float(action_bytes.decode('utf-8'))
                    self.get_logger().info(f"Received action: {force:.4f}")

                    # CRITICAL: Synchronization
                    # Clear old state and publish new action
                    with self.state_lock:
                        self.latest_state = None
                    action_msg = Float64(data=force)
                    self.action_publisher.publish(action_msg)
                    # Wait for the new state to arrive from the /state topic
                    timeout_start = time.time()
                    new_state_received = False
                    while not new_state_received and (time.time() - timeout_start) < 2.0:
                        with self.state_lock:
                            if self.latest_state is not None:
                                new_state_received = True
                        if not new_state_received:
                            time.sleep(0.001) # Yield thread

                    # Return the state (REP)
                    if new_state_received:
                        pos, vel, sec, nanosec = self.latest_state
                        reply_payload = f"{pos},{vel},{sec},{nanosec}".encode('utf-8')
                    else:
                        self.get_logger().warn("State timeout! Replying with 0 state.")
                        reply_payload = b"0.0,0.0,0,0" # Fail safe
                    
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
