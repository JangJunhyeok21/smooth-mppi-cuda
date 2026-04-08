import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool  
import subprocess
import time
import csv
import itertools
import math
import os
import signal
from ament_index_python.packages import get_package_share_directory

class MPPIOptimizer(Node):
    def __init__(self):
        super().__init__('mppi_optimizer')
        
        # 1. 테스트할 파라미터 범위 설정 (Grid Search)
        self.q_v_list = [1.3, 1.5, 1.7]
        self.q_dist_list = [0.0]
        self.q_du_list = [0.5, 0.7, 1.0]
        self.q_steer_list = [0.3, 0.5, 0.7]
        self.q_lat_g_list = [150.0, 200.0]
        self.q_collision_list = [200.0, 150.0]
        
        self.param_combinations = list(itertools.product(
            self.q_v_list, self.q_dist_list, self.q_du_list, 
            self.q_steer_list, self.q_lat_g_list, self.q_collision_list
        ))
        self.get_logger().info(f"Total Combinations to run: {len(self.param_combinations)}")
        
        # 원본 yaml 경로 캐싱
        self.base_yaml_path = os.path.join(
            get_package_share_directory("smppi_cuda_controller"),
            "config",
            "params.yaml"
        )
        
        self.odom_sub = self.create_subscription(Odometry, '/odom0', self.odom_callback, 10)
        self.collision_sub = self.create_subscription(Bool, '/collision0', self.collision_callback, 10)
        
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd0', 10)
        
        self.results = []
        self.current_run = 0
        
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_v = 0.0
        self.start_time = 0.0
        self.is_running = False
        self.has_crashed = False  
        self.max_distance = 0.0
        self.mppi_process = None
        self.mppi_log_file = None

        # 시작 포즈 및 완료 판정 기준
        self.start_x = -14.6
        self.start_y = -4.83
        self.start_yaw = 1.570796
        self.min_lap_distance = 5.0

        self.reset_pending = False
        self.reset_deadline = 0.0

        self.timer = self.create_timer(1.0, self.optimization_loop)

    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        self.car_v = msg.twist.twist.linear.x

    def collision_callback(self, msg):
        if msg.data == True: 
            self.has_crashed = True
            self.get_logger().warn("Simulator reported a CRASH!")

    def reset_simulation(self):
        self.get_logger().info("Resetting Simulation...")
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_msg)

        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = 'map'
        init_pose.header.stamp = self.get_clock().now().to_msg()
        init_pose.pose.pose.position.x = self.start_x
        init_pose.pose.pose.position.y = self.start_y
        init_pose.pose.pose.position.z = 0.0
        init_pose.pose.pose.orientation.z = math.sin(self.start_yaw * 0.5)
        init_pose.pose.pose.orientation.w = math.cos(self.start_yaw * 0.5)
        self.init_pose_pub.publish(init_pose)
        
        self.has_crashed = False
        self.max_distance = 0.0
        self.reset_pending = True
        self.reset_deadline = time.time() + 1.0

    def start_mppi_node(self, q_v, q_dist, q_du, q_steer, q_lat_g, q_collision):
        """🚨 ros2 run으로 제어기만 단독 실행. 베이스 yaml 위에 최적화 변수만 덮어씌움"""
        cmd = [
            "ros2", "run", "smppi_cuda_controller", "smppi_node",
            "--ros-args",
            "--params-file", self.base_yaml_path,  # 기본 차량 세팅(D_f, mass 등)은 여기서 로드
            "-p", f"q_v:={q_v}",                   # 아래 6개 변수만 실시간 덮어쓰기
            "-p", f"q_dist:={q_dist}",
            "-p", f"q_du:={q_du}",
            "-p", f"q_steer:={q_steer}",
            "-p", f"q_lat_g:={q_lat_g}",
            "-p", f"q_collision:={q_collision}",
            "-p", "use_mcl_pose:=False"            # 시뮬레이터 모드 강제
        ]
        
        self.get_logger().info(f"Run {self.current_run + 1}: q_v={q_v}, lat_g={q_lat_g}, col={q_collision}")
        
        os.makedirs("result", exist_ok=True)
        log_path = f"result/mppi_node_run_{self.current_run + 1}.log"
        self.mppi_log_file = open(log_path, "w")
        
        self.mppi_process = subprocess.Popen(
            cmd,
            stdout=self.mppi_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        self.start_time = time.time()
        self.is_running = True

    def stop_mppi_node(self):
        """제어기 노드 강제 종료"""
        if self.mppi_process:
            if self.mppi_process.poll() is None:
                try:
                    os.killpg(self.mppi_process.pid, signal.SIGTERM)
                    self.mppi_process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self.mppi_process.pid, signal.SIGKILL)
                    self.mppi_process.wait()
                except ProcessLookupError:
                    pass
            self.mppi_process = None
            
        if self.mppi_log_file:
            self.mppi_log_file.close()
            self.mppi_log_file = None
        self.is_running = False

    def optimization_loop(self):
        if self.current_run >= len(self.param_combinations):
            self.save_results()
            self.get_logger().info("Optimization Finished!")
            rclpy.shutdown()
            return

        if not self.is_running:
            if not self.reset_pending:
                self.reset_simulation()
                return
            if time.time() < self.reset_deadline:
                return
            self.reset_pending = False
            q_v, q_dist, q_du, q_steer, q_lat_g, q_col = self.param_combinations[self.current_run]
            self.start_mppi_node(q_v, q_dist, q_du, q_steer, q_lat_g, q_col)

        else:
            elapsed_time = time.time() - self.start_time
            distance_from_start = math.hypot(self.car_x - self.start_x, self.car_y - self.start_y)
            self.max_distance = max(self.max_distance, distance_from_start)

            is_crashed = self.has_crashed
            is_lap_done = (elapsed_time > 10.0) and (distance_from_start < 2.0) and (self.max_distance > self.min_lap_distance)
            is_timeout = elapsed_time > 60.0

            if is_crashed or is_lap_done or is_timeout:
                self.stop_mppi_node()
                
                status = "Finished" if is_lap_done else ("Crashed" if is_crashed else "Timeout")
                q_v, q_dist, q_du, q_steer, q_lat_g, q_col = self.param_combinations[self.current_run]
                
                self.results.append({
                    'q_v': q_v, 'q_dist': q_dist, 'q_du': q_du, 'q_steer': q_steer,
                    'q_lat_g': q_lat_g, 'q_collision': q_col,
                    'status': status, 'lap_time': elapsed_time if is_lap_done else 999.0,
                    'max_distance': self.max_distance
                })
                
                self.get_logger().info(f"Ended: {status}, Time: {elapsed_time:.2f}s, Dist: {self.max_distance:.2f}m")
                self.current_run += 1

    def save_results(self):
        with open('result/mppi_optimization_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['q_v', 'q_dist', 'q_du', 'q_steer', 'q_lat_g', 'q_collision', 'status', 'lap_time', 'max_distance']
            )
            writer.writeheader()
            writer.writerows(self.results)
        self.get_logger().info("Results saved to result/mppi_optimization_results.csv")

def main(args=None):
    rclpy.init(args=args)
    node = MPPIOptimizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_mppi_node()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()