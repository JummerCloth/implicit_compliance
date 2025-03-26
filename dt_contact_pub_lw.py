#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time


class ContactClassifierNode(Node):
    def __init__(self):
        super().__init__('contact_classifier_node')
        
        # Create publishers
        self.contact_publisher = self.create_publisher(Int8, '/dt/contact', 10)
        self.image_publisher = self.create_publisher(CompressedImage, '/dt/stream', 10)
        
        # Create timer for camera processing (10Hz)
        self.timer = self.create_timer(0.1, self.process_camera_callback)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open video stream.")
            rclpy.shutdown()
            return
            
        # Set camera parameters
        self.set_camera_resolution(self.cap, 640, 480)
        time.sleep(1)  # Allow camera to warm up
        
        # Load reference images
        self.contact_images = []
        self.no_contact_images = []
        self.load_reference_images()
        
        # Initialize prediction status
        self.last_status = None
        
        self.get_logger().info("Contact classifier node started. Publishing to /dt/contact and /dt/stream")
    
    def load_reference_images(self):
        # Paths to the reference images
        contact_reference_image_paths = [
            "contact_reference/images/image_20250305_005203.jpg",
            "contact_reference/images/image_20250305_005206.jpg",
            "contact_reference/images/image_20250305_005207.jpg",
            "contact_reference/images/image_20250305_005209.jpg",
            "contact_reference/images/image_20250305_005210.jpg",
        ]

        no_contact_reference_image_paths = [
            "contact_reference/images/image_20250304_233311.jpg",
            "contact_reference/images/image_20250304_233312.jpg",
            "contact_reference/images/image_20250304_233313.jpg",
            "contact_reference/images/image_20250304_233314.jpg",
            "contact_reference/images/image_20250304_233315.jpg"
        ]

        # Load contact reference images
        try:
            for path in contact_reference_image_paths:
                img = cv2.imread(path)
                if img is None:
                    self.get_logger().warn(f"Could not load image: {path}")
                    continue
                
                # Preprocess reference image
                processed_img = self.center_crop_and_resize(img)
                self.contact_images.append(processed_img)
            
            self.get_logger().info(f"Loaded {len(self.contact_images)} contact reference images")
        except Exception as e:
            self.get_logger().error(f"Error loading contact reference images: {e}")

        # Load no-contact reference images
        try:
            for path in no_contact_reference_image_paths:
                img = cv2.imread(path)
                if img is None:
                    self.get_logger().warn(f"Could not load image: {path}")
                    continue
                
                # Preprocess reference image
                processed_img = self.center_crop_and_resize(img)
                self.no_contact_images.append(processed_img)
            
            self.get_logger().info(f"Loaded {len(self.no_contact_images)} no-contact reference images")
        except Exception as e:
            self.get_logger().error(f"Error loading no-contact reference images: {e}")

        # Check if we have enough reference images to proceed
        if len(self.contact_images) == 0 or len(self.no_contact_images) == 0:
            self.get_logger().error("Not enough reference images were loaded")
            rclpy.shutdown()
    
    def process_camera_callback(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().error("Can't receive frame. Exiting...")
            rclpy.shutdown()
            return

        # Publish the compressed image
        self.publish_compressed_image(frame)
            
        # Preprocess the frame
        processed_frame = self.center_crop_and_resize(frame)
        
        # Calculate differences to all contact reference images
        contact_diffs = []
        for ref_img in self.contact_images:
            diff = self.image_difference(processed_frame, ref_img)
            contact_diffs.append(diff)
        
        # Calculate differences to all no-contact reference images
        no_contact_diffs = []
        for ref_img in self.no_contact_images:
            diff = self.image_difference(processed_frame, ref_img)
            no_contact_diffs.append(diff)
        
        # Get the minimum difference for each category
        min_contact_diff = min(contact_diffs)
        min_no_contact_diff = min(no_contact_diffs)
        
        # Get the average difference for each category
        avg_contact_diff = sum(contact_diffs) / len(contact_diffs)
        avg_no_contact_diff = sum(no_contact_diffs) / len(no_contact_diffs)
        
        # Apply a bias factor to favor contact detection (makes the threshold lower)
        # A bias of 0.9 means contact diff is multiplied by 0.9, making it 10% more likely to be detected
        bias_factor = 0.8  # Adjust this value between 0.8-0.95 to control sensitivity
        
        # Determine classification based on minimum difference with bias
        if min_contact_diff * bias_factor < min_no_contact_diff:
            status = "Contact"
            contact_state = 1  # Contact = 1
            confidence = 1.0 - ((min_contact_diff * bias_factor) / (min_contact_diff * bias_factor + min_no_contact_diff))
        else:
            status = "No Contact"
            contact_state = 0  # No Contact = 0
            confidence = 1.0 - (min_no_contact_diff / (min_contact_diff * bias_factor + min_no_contact_diff))
        
        # Publish contact state to ROS topic
        msg = Int8()
        msg.data = contact_state
        self.contact_publisher.publish(msg)
        
        # Only log if status changes or significant change in confidence
        if self.last_status != status:
            self.get_logger().info(f"Publishing: {status} ({contact_state}) [confidence: {confidence:.2f}]")
            self.get_logger().debug(f"Min contact diff: {min_contact_diff:.2f}, Min no-contact diff: {min_no_contact_diff:.2f}")
            self.get_logger().debug(f"Avg contact diff: {avg_contact_diff:.2f}, Avg no-contact diff: {avg_no_contact_diff:.2f}")
            self.last_status = status
    
    def publish_compressed_image(self, frame):
        """
        Publish a compressed image to the /dt/stream topic
        """
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        
        # Resize image to reduce bandwidth (optional)
        # You can adjust the size based on your requirements
        resized_frame = cv2.resize(frame, (320, 240)) 
        
        # Compress the image with JPEG
        # You can adjust the quality (0-100) to balance between size and quality
        # Lower values = smaller file size but lower quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, jpeg_img = cv2.imencode('.jpg', resized_frame, encode_param)
        
        # Convert to bytes and publish
        msg.data = np.array(jpeg_img).tobytes()
        self.image_publisher.publish(msg)
    
    def set_camera_resolution(self, cap, width, height):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_EXPOSURE, -50)

    def center_crop_and_resize(self, img, target_size=(224, 224)):
        """
        Crops the center of a 640x480 image to 480x480 and then resizes it to target_size.
        """
        h, w = img.shape[:2]  # Get image dimensions
        
        # Calculate cropping box for center crop
        crop_size = min(h, w)  # Make it square
        start_x = (w - crop_size) // 2  # Center horizontally
        start_y = (h - crop_size) // 2  # Center vertically

        cropped_img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        resized_img = cv2.resize(cropped_img, target_size)

        return resized_img

    def image_difference(self, img1, img2):
        """
        Calculate the mean absolute difference between two images.
        Lower values indicate more similarity.
        """
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale for simpler comparison
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            
        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1_gray, img2_gray)
        
        # Return mean difference (lower is more similar)
        return np.mean(diff)
    
    def __del__(self):
        # Release the camera when node is destroyed
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            

def main(args=None):
    rclpy.init(args=args)
    
    node = ContactClassifierNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly")
    except Exception as e:
        node.get_logger().error(f"Error occurred: {e}")
    finally:
        # Destroy the node explicitly
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
