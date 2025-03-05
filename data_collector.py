#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Int8
import csv
import argparse
import time
import threading
import os
from datetime import datetime

class DataCollector(Node):
    def __init__(self, output_index, output_dir='./data'):
        super().__init__('data_collector')
        
        # Store the output index and create output directory if it doesn't exist
        self.output_index = output_index
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create CSV files with headers
        self.joint_states_file = f"{self.output_dir}/joint_states_{self.output_index}.csv"
        self.contact_file = f"{self.output_dir}/contact_{self.output_index}.csv"
        
        # Create an image directory for storing the compressed images
        self.image_dir = f"{self.output_dir}/images_{self.output_index}"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            
        # Create an index file for the images
        self.image_index_file = f"{self.output_dir}/image_index_{self.output_index}.csv"
        
        # Initialize CSV files with headers
        with open(self.joint_states_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'leg_front_r_1_position', 'leg_front_r_2_position', 'leg_front_r_3_position', 
                            'leg_front_r_1_velocity', 'leg_front_r_2_velocity', 'leg_front_r_3_velocity', 
                            'leg_front_r_1_effort', 'leg_front_r_2_effort', 'leg_front_r_3_effort'])
            
        with open(self.contact_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'contact_value'])
            
        with open(self.image_index_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'filename', 'format'])
        
        # Set up subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
            
        self.contact_sub = self.create_subscription(
            Int8,
            '/dt/contact',
            self.contact_callback,
            10)
        
        # For image, we'll use a timer to throttle to 3 Hz
        self.last_image = None
        self.image_lock = threading.Lock()
        
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/dt/stream',
            self.image_callback_store,
            10)
            
        # Timer for saving images at 3Hz
        self.create_timer(1.0/3.0, self.image_timer_callback)
        
        self.get_logger().info(f'Data collector initialized. Saving to files with index {output_index}')
    
    def joint_states_callback(self, msg):
        # Find indices for the joints we're interested in
        indices = {}
        for joint in ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']:
            if joint in msg.name:
                indices[joint] = msg.name.index(joint)
            else:
                indices[joint] = None
        
        # Only log if all required joints are present
        if all(idx is not None for idx in indices.values()):
            timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
            
            # Extract the position, velocity, and effort for each joint
            data_row = [timestamp]
            
            # Add positions
            for joint in ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']:
                idx = indices[joint]
                data_row.append(msg.position[idx])
            
            # Add velocities
            for joint in ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']:
                idx = indices[joint]
                data_row.append(msg.velocity[idx] if msg.velocity else 0.0)
            
            # Add efforts
            for joint in ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']:
                idx = indices[joint]
                data_row.append(msg.effort[idx] if msg.effort else 0.0)
            
            # Write to file
            with open(self.joint_states_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
    
    def contact_callback(self, msg):
        timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        
        # Write to file
        with open(self.contact_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, msg.data])
    
    def image_callback_store(self, msg):
        # Store the latest image for use in the timer callback
        with self.image_lock:
            self.last_image = msg
    
    def image_timer_callback(self):
        # This is called at 3Hz to save image data
        with self.image_lock:
            if self.last_image is not None:
                timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
                
                # Generate unique filename using timestamp
                filename = f"img_{timestamp:.6f}.jpg"
                filepath = os.path.join(self.image_dir, filename)
                
                # Save the raw compressed image data directly to a file
                with open(filepath, 'wb') as f:
                    f.write(bytes(self.last_image.data))
                
                # Update the index CSV with metadata
                with open(self.image_index_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        filename,
                        self.last_image.format
                    ])
                
                # Clear to avoid duplicates
                self.last_image = None

def main(args=None):
    parser = argparse.ArgumentParser(description='ROS2 Data Collector')
    parser.add_argument('--index', type=str, required=True, help='Output file index')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')
    
    args = parser.parse_args()
    
    rclpy.init()
    data_collector = DataCollector(args.index, args.output_dir)
    
    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        data_collector.get_logger().info('Data collection stopped by user')
    finally:
        data_collector.get_logger().info('Shutting down...')
        data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()