import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32
from std_msgs.msg import Int8
from collections import deque
import numpy as np

np.set_printoptions(precision=3, suppress=True)

Kp = 3
Kd = 0.1

class InverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics')
        self.joint_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.joint_subscription  # prevent unused variable warning

        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            '/forward_command_controller/commands',
            10
        )

        self.contact_subscription = self.create_subscription(
            Int8,
            '/dt/contact',
            self.contact_callback,
            10
        )
        self.contact_subscription  # prevent unused variable warning

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        self.ik_timer_period = 1.0 / 20   # 10 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None
        self.stop_leg = False
        
        # Speed factor for motion (lower value = slower motion)
        self.speed_factor = 0.4  # Adjust this value to slow down or speed up (1.0 is original speed)

        self.ee_triangle_positions = np.array([
            [0.05, 0.0, -0.12],  # Touchdown
            [-0.05, 0.0, -0.12], # Liftoff
            [0.0, 0.0, -0.06]    # Mid-swing
        ])

        center_to_rf_hip = np.array([0.07500, -0.08350, 0])
        self.ee_triangle_positions = self.ee_triangle_positions + center_to_rf_hip
        self.current_target = 0
        self.t = 0
        
        # Track which part of the triangle motion we're in
        self.current_motion_segment = 0  # 0: touchdown to liftoff, 1: liftoff to mid-swing, 2: mid-swing to touchdown
        
        self.get_logger().info(f"Inverse Kinematics node started with speed factor: {self.speed_factor}")

    def listener_callback(self, msg):
        joints_of_interest = ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']
        self.joint_positions = np.array([msg.position[msg.name.index(joint)] for joint in joints_of_interest])
        self.joint_velocities = np.array([msg.velocity[msg.name.index(joint)] for joint in joints_of_interest])

    def contact_callback(self, msg):
        # Only stop the leg if we're in the first segment (touchdown to liftoff)
        if self.current_motion_segment == 0 and msg.data == 1:
            self.get_logger().info(f"Contact detected in touchdown-to-liftoff segment: stopping leg")
            self.stop_leg = True
        elif msg.data == 0:
            # Always allow the leg to restart moving when contact is released
            if self.stop_leg:
                self.get_logger().info(f"Contact released: resuming leg motion")
            self.stop_leg = False

    def forward_kinematics(self, theta1, theta2, theta3):
        def rotation_x(angle):
           return np.array([
               [1, 0, 0, 0],
               [0, np.cos(angle), -np.sin(angle), 0],
               [0, np.sin(angle), np.cos(angle), 0],
               [0, 0, 0, 1]
           ])

        def rotation_y(angle):
           return np.array([
               [np.cos(angle), 0, np.sin(angle), 0],
               [0, 1, 0, 0],
               [-np.sin(angle), 0, np.cos(angle), 0],
               [0, 0, 0, 1]
           ])
      
        def rotation_z(angle):
           return np.array([
               [np.cos(angle), -np.sin(angle), 0, 0],
               [np.sin(angle), np.cos(angle), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]
           ])

        def translation(x, y, z):
           return np.array([
               [1, 0, 0, x],
               [0, 1, 0, y],
               [0, 0, 1, z],
               [0, 0, 0, 1]
           ])

        T_0_1 = translation(0.07500, -0.0445, 0) @ rotation_x(1.57080) @ rotation_z(theta1)
        T_1_2 = translation(0, 0, 0.039) @ rotation_y(-np.pi / 2) @ rotation_z(theta2)
        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(np.pi / 2) @ rotation_z(theta3)
        T_3_ee = translation(0.06231, -0.06216, 0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee

        joint_pos = np.array([0, 0, 0, 1])
        joint_pos[:3] = self.joint_positions
        end_effector_position = T_0_ee @ joint_pos.T
        end_effector_position = (end_effector_position[:3] / end_effector_position[-1]).T
        return end_effector_position

    def inverse_kinematics(self, target_ee, initial_guess=[0, 0, 0], arg = "GD"):
        def cost_function(theta):
            theta1, theta2, theta3 = theta
            gt_ee = self.forward_kinematics(theta1, theta2, theta3)
            return np.sum((gt_ee - target_ee)**2), np.linalg.norm(gt_ee - target_ee)

        def gradient(theta, epsilon=1e-3):
            gradient = np.zeros([3,])
            for i, t in enumerate(theta):
                theta_min = [theta[0] - (i == 0)* epsilon, theta[1] - (i == 1)* epsilon, theta[2] - (i == 2) * epsilon]
                theta_max = [theta[0] + (i == 0)* epsilon, theta[1] + (i == 1)* epsilon, theta[2] + (i == 2) * epsilon]
                gradient[i] = (cost_function(theta_max)[0] - cost_function(theta_min)[0]) / (2 * epsilon)
            return gradient

        theta = np.array(initial_guess)
        learning_rate = 5
        max_iterations = 50
        tolerance = 1e-2

        cost_l = []
        for _ in range(max_iterations):
            grad = gradient(theta)
            if arg == "GD":
                theta -= learning_rate * grad
            elif arg == "Newton":
                raise KeyboardInterrupt("You're bad")

            cost, l1 = cost_function(theta)
            cost_l.append(cost)
            if l1.mean() < tolerance:
                break

        return theta

    def interpolate_triangle(self, t):
        vertex1, vertex2, vertex3 = self.ee_triangle_positions
        t_res = t % 3
        
        # Update the current motion segment based on t_res
        if 0 <= t_res <= 1:
            self.current_motion_segment = 0  # touchdown to liftoff
            return vertex1 + t_res * (vertex2 - vertex1)
        elif 1 < t_res <= 2:
            self.current_motion_segment = 1  # liftoff to mid-swing
            return vertex2 + (t_res-1) * (vertex3 - vertex2)
        else:
            self.current_motion_segment = 2  # mid-swing to touchdown
            return vertex3 + (t_res-2) * (vertex1 - vertex3)
    
    def ik_timer_callback(self):
        if self.joint_positions is not None and not self.stop_leg:
            target_ee = self.interpolate_triangle(self.t)
            self.target_joint_positions = self.inverse_kinematics(target_ee, self.joint_positions)
            current_ee = self.forward_kinematics(*self.joint_positions)
            
            # Update time with speed factor
            # Original line: self.t += self.ik_timer_period * 2
            self.t += self.ik_timer_period * 2 * self.speed_factor
            
            # Periodically log the current segment for monitoring
            if int(self.t * 100) % 100 == 0:  # Log occasionally
                self.get_logger().info(f'Current motion segment: {self.current_motion_segment}, t: {self.t % 3:.2f}')

    def pd_timer_callback(self):
        if self.target_joint_positions is not None and not self.stop_leg:
            command_msg = Float64MultiArray()
            command_msg.data = self.target_joint_positions.tolist()
            self.command_publisher.publish(command_msg)

def main():
    rclpy.init()
    inverse_kinematics = InverseKinematics()
    
    try:
        rclpy.spin(inverse_kinematics)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        zero_torques = Float64MultiArray()
        zero_torques.data = [0.0, 0.0, 0.0]
        inverse_kinematics.command_publisher.publish(zero_torques)
        
        inverse_kinematics.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()