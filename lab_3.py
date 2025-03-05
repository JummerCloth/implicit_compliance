import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
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

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        self.ik_timer_period = 1.0 / 20   # 10 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None

        self.ee_triangle_positions = np.array([
            [0.05, 0.0, -0.12],  # Touchdown
            [-0.05, 0.0, -0.12], # Liftoff
            [0.0, 0.0, -0.06]    # Mid-swing
        ])

        center_to_rf_hip = np.array([0.07500, -0.08350, 0])
        self.ee_triangle_positions = self.ee_triangle_positions + center_to_rf_hip
        self.current_target = 0
        self.t = 0

    def listener_callback(self, msg):
        joints_of_interest = ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']
        self.joint_positions = np.array([msg.position[msg.name.index(joint)] for joint in joints_of_interest])
        self.joint_velocities = np.array([msg.velocity[msg.name.index(joint)] for joint in joints_of_interest])

    def forward_kinematics(self, theta1, theta2, theta3):
        ################################################################################################
        # TODO: paste lab 2 forward kinematics here
        ################################################################################################
        def rotation_x(angle):
           # rotation about the x-axis implemented for you
           return np.array([
               [1, 0, 0, 0],
               [0, np.cos(angle), -np.sin(angle), 0],
               [0, np.sin(angle), np.cos(angle), 0],
               [0, 0, 0, 1]
           ])


        def rotation_y(angle):
           ## TODO: Implement the rotation matrix about the y-axis
           return np.array([
               [np.cos(angle), 0, np.sin(angle), 0],
               [0, 1, 0, 0],
               [-np.sin(angle), 0, np.cos(angle), 0],
               [0, 0, 0, 1]
           ])
      
        def rotation_z(angle):
           ## TODO: Implement the rotation matrix about the z-axis
           return np.array([
               [np.cos(angle), -np.sin(angle), 0, 0],
               [np.sin(angle), np.cos(angle), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]
           ])


        def translation(x, y, z):
           ## TODO: Implement the translation matrix
           return np.array([
               [1, 0, 0, x],
               [0, 1, 0, y],
               [0, 0, 1, z],
               [0, 0, 0, 1]
           ])
           None


        # T_0_1 (base_link to leg_front_r_1)
        T_0_1 = translation(0.07500, -0.0445, 0) @ rotation_x(1.57080) @ rotation_z(theta1)


        # T_1_2 (leg_front_r_1 to leg_front_r_2)
        ## TODO: Implement the transformation matrix from leg_front_r_1 to leg_front_r_2
        T_1_2 = translation(0, 0, 0.039) @ rotation_y(-np.pi / 2) @ rotation_z(theta2)


        # T_2_3 (leg_front_r_2 to leg_front_r_3)
        ## TODO: Implement the transformation matrix from leg_front_r_2 to leg_front_r_3
        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(np.pi / 2) @ rotation_z(theta3)


        # T_3_ee (leg_front_r_3 to end-effector)
        T_3_ee = translation(0.06231, -0.06216, 0.018)


        # TODO: Compute the final transformation. T_0_ee is a concatenation of the previous transformation matrices
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee


        # TODO: Extract the end-effector position. The end effector position is a 3x3 matrix (not in homogenous coordinates)
        # project to homogenous space
        joint_pos = np.array([0, 0, 0, 1])
        joint_pos[:3] = self.joint_positions
        end_effector_position = T_0_ee @ joint_pos.T
        # project back to euclidean space
        end_effector_position = (end_effector_position[:3] / end_effector_position[-1]).T
        return end_effector_position

    def inverse_kinematics(self, target_ee, initial_guess=[0, 0, 0], arg = "GD"): # can specity arg = "Newton"
        def cost_function(theta):
            # Compute the cost function and the L1 norm of the error
            # return the cost and the L1 norm of the error
            ################################################################################################
            # TODO: Implement the cost function
            ################################################################################################
            theta1, theta2, theta3 = theta
            gt_ee = self.forward_kinematics(theta1, theta2, theta3)
            return np.sum((gt_ee - target_ee)**2), np.linalg.norm(gt_ee - target_ee)


        def gradient(theta, epsilon=1e-3):
            # Compute the gradient of the cost function using finite differences
            ################################################################################################
            # TODO: Implement the gradient computation
            ################################################################################################
            gradient = np.zeros([3,])
            for i, t in enumerate(theta):
                theta_min = [theta[0] - (i == 0)* epsilon, theta[1] - (i == 1)* epsilon, theta[2] - (i == 2) * epsilon]
                theta_max = [theta[0] + (i == 0)* epsilon, theta[1] + (i == 1)* epsilon, theta[2] + (i == 2) * epsilon]
                gradient[i] = (cost_function(theta_max)[0] - cost_function(theta_min)[0]) / (2 * epsilon)
            return gradient

        theta = np.array(initial_guess)
        learning_rate = 5 # TODO: Set the learning rate
        max_iterations = 50 # TODO: Set the maximum number of iterations
        tolerance = 1e-2 # TODO: Set the tolerance for the L1 norm of the error

        cost_l = []
        for _ in range(max_iterations):
            grad = gradient(theta)
            # Update the theta (parameters) using the gradient and the learning rate
            ################################################################################################
            # TODO: Implement the gradient update
            # TODO (BONUS): Implement the (quasi-)Newton's method for faster convergence
            if arg == "GD":
                theta -= learning_rate * grad
            elif arg == "Newton":
                raise KeyboardInterrupt("You're bad")
            ################################################################################################

            cost, l1 = cost_function(theta)
            cost_l.append(cost)
            if l1.mean() < tolerance:
                break

        print(f'Cost: {cost_l}')

        return theta

    def interpolate_triangle(self, t):
        # Intepolate between the three triangle positions in the self.ee_triangle_positions
        # based on the current time t
        ################################################################################################
        # TODO: Implement the interpolation function
        ################################################################################################
        vertex1, vertex2, vertex3 = self.ee_triangle_positions
        t_res = t % 3
        if 0 <= t_res <= 1:
            return vertex1 + t_res * (vertex2 - vertex1)
        elif 1 < t_res <= 2:
            return vertex2 + (t_res-1) * (vertex3 - vertex2)
        else:
            return vertex3 + (t_res-2) * (vertex1 - vertex3)
    
    def ik_timer_callback(self):
        if self.joint_positions is not None:
            target_ee = self.interpolate_triangle(self.t)
            self.target_joint_positions = self.inverse_kinematics(target_ee, self.joint_positions)
            current_ee = self.forward_kinematics(*self.joint_positions)

            # update the current time for the triangle interpolation
            ################################################################################################
            # TODO: Implement the time update
            ################################################################################################
            self.t += self.ik_timer_period * 2
            
            self.get_logger().info(f'Target EE: {target_ee}, Current EE: {current_ee}, Target Angles: {self.target_joint_positions}, Target Angles to EE: {self.forward_kinematics(*self.target_joint_positions)}, Current Angles: {self.joint_positions}')

    def pd_timer_callback(self):
        if self.target_joint_positions is not None:
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
        # Send zero torques
        zero_torques = Float64MultiArray()
        zero_torques.data = [0.0, 0.0, 0.0]
        inverse_kinematics.command_publisher.publish(zero_torques)
        
        inverse_kinematics.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
