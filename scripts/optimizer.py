import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool  # 시뮬레이터의 충돌 메시지 타입에 맞게 변경 (예: String, Empty 등)
import subprocess
import time
import csv
import itertools
import math
import os
import signal

class MPPIOptimizer(Node):
    def __init__(self):
        super().__init__('mppi_optimizer')
        
        # 1. 테스트할 파라미터 범위 설정 (Grid Search)
        self.q_v_list = [1.8, 1.9, 2.0, 2.1, 2.2]  # 속도 비용 가중치
        self.q_dist_list = [1.3, 1.4, 1.5, 1.6, 1.7]  # 중심선 거리 비용 가중치
        self.q_du_list = [0.0, 0.4, 0.8, 1.2]  # 조작 변화량 비용 가중치
        self.q_steer_list = [0.0, 0.3, 0.6, 0.9]  # 조향각 비용 가중치
        self.param_combinations = list(itertools.product(self.q_v_list, self.q_dist_list, self.q_du_list, self.q_steer_list))
        
        # ROS 2 통신 설정
        self.odom_sub = self.create_subscription(Odometry, '/odom0', self.odom_callback, 10)
        
        # [수정] 시뮬레이터의 충돌 토픽 구독 (토픽명 '/collisions' 및 타입은 환경에 맞게 수정)
        self.collision_sub = self.create_subscription(Bool, '/collision0', self.collision_callback, 10)
        
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd0', 10)
        
        self.results = []
        self.current_run = 0
        
        # 상태 변수
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_v = 0.0
        self.start_time = 0.0
        self.is_running = False
        self.has_crashed = False  # 충돌 상태 플래그
        self.max_distance = 0.0
        self.mppi_process = None
        self.mppi_log_file = None

        # 시작 포즈 및 완료 판정 기준
        self.start_x = -14.6
        self.start_y = -4.83
        self.start_yaw = 1.570796
        self.min_lap_distance = 5.0

        # 리셋 상태 관리
        self.reset_pending = False
        self.reset_deadline = 0.0

        # 메인 최적화 루프 타이머
        self.timer = self.create_timer(1.0, self.optimization_loop)

    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        self.car_v = msg.twist.twist.linear.x

    def collision_callback(self, msg):
        # [수정] 충돌 메시지가 들어오면 플래그를 True로 변경
        # 만약 메시지가 Bool 타입이고 True일 때 충돌이라면 아래와 같이 작성합니다.
        if msg.data == True: 
            self.has_crashed = True
            self.get_logger().warn("Simulator reported a CRASH!")

    def reset_simulation(self):
        """차량을 출발선으로 되돌리고 상태 초기화"""
        self.get_logger().info("Resetting Simulation...")
        
        # 정지 명령 전송
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_msg)

        # 트랙의 시작 위치(원점)로 텔레포트
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = 'map'
        init_pose.header.stamp = self.get_clock().now().to_msg()
        init_pose.pose.pose.position.x = self.start_x
        init_pose.pose.pose.position.y = self.start_y
        init_pose.pose.pose.position.z = 0.0
        init_pose.pose.pose.orientation.z = math.sin(self.start_yaw * 0.5)
        init_pose.pose.pose.orientation.w = math.cos(self.start_yaw * 0.5)
        self.init_pose_pub.publish(init_pose)
        
        # 충돌 플래그 및 거리 초기화
        self.has_crashed = False
        self.max_distance = 0.0
        self.reset_pending = True
        self.reset_deadline = time.time() + 1.0

    def start_mppi_node(self, q_v, q_dist, q_du, q_steer):
        """서브프로세스로 파라미터를 주입하여 C++ 제어기 노드 실행"""
        cmd = [
            "ros2", "run", "cuda_mppi_controller", "cuda_mppi_node",
            "--ros-args",
            "-p", f"q_v:={q_v}",
            "-p", f"q_dist:={q_dist}",
            "-p", f"q_du:={q_du}",
            "-p", f"q_steer:={q_steer}"
        ]
        self.get_logger().info(f"Starting Node with q_v={q_v}, q_dist={q_dist}, q_du={q_du}, q_steer={q_steer}")
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
        """C++ 제어기 노드 강제 종료"""
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
            # 새로운 주행 시작
            if not self.reset_pending:
                self.reset_simulation()
                return
            if time.time() < self.reset_deadline:
                return
            self.reset_pending = False
            q_v, q_dist, q_du, q_steer = self.param_combinations[self.current_run]
            self.start_mppi_node(q_v, q_dist, q_du, q_steer)

        else:
            # 주행 모니터링
            elapsed_time = time.time() - self.start_time
            distance_from_start = math.hypot(self.car_x - self.start_x, self.car_y - self.start_y)
            self.max_distance = max(self.max_distance, distance_from_start)

            # [종료 조건 1] 시뮬레이터에서 보내준 충돌 플래그 확인
            is_crashed = self.has_crashed
            
            # [종료 조건 2] 완주 감지 (10초 이후 출발선 2m 이내 복귀 등 트랙 모양에 맞게 설정)
            is_lap_done = (elapsed_time > 10.0) and (distance_from_start < 2.0) and (self.max_distance > self.min_lap_distance)

            # [종료 조건 3] 타임아웃
            is_timeout = elapsed_time > 60.0

            if is_crashed or is_lap_done or is_timeout:
                self.stop_mppi_node()
                
                status = "Finished" if is_lap_done else ("Crashed" if is_crashed else "Timeout")
                q_v, q_dist, q_du, q_steer = self.param_combinations[self.current_run]
                
                # 결과 저장
                self.results.append({
                    'q_v': q_v,
                    'q_dist': q_dist,
                    'q_du': q_du,
                    'q_steer': q_steer,
                    'status': status,
                    'lap_time': elapsed_time if is_lap_done else 999.0,
                    'max_distance': self.max_distance
                })
                
                self.get_logger().info(f"Run {self.current_run+1} Ended: {status}, Time: {elapsed_time:.2f}s, Dist: {self.max_distance:.2f}m")
                self.current_run += 1

    def save_results(self):
        with open('result/mppi_optimization_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['q_v', 'q_dist', 'q_du', 'q_steer', 'status', 'lap_time', 'max_distance']
            )
            writer.writeheader()
            writer.writerows(self.results)
        self.get_logger().info("Results saved to mppi_optimization_results.csv")

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