import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from f1_msgs.msg import CollisionEvent
import subprocess
import time
import csv
import itertools
import math
import os
import signal
from ament_index_python.packages import get_package_share_directory

CSV_FIELDNAMES = [
    'q_v', 'q_dist', 'q_du', 'q_steer', 'q_lat_g', 'q_collision', 'q_progress', 'q_escape_vel',
    'lat_g_soft_limit', 'longitudinal_accel_soft_limit',
    'status', 'lap_time', 'max_distance', 'collision_source',
]

# f1_msgs/CollisionEvent.source 값 -> 사람이 읽을 문자열
_COLLISION_SOURCE_NAMES = {
    CollisionEvent.NONE: 'none',
    CollisionEvent.WALL: 'wall',
    CollisionEvent.VEHICLE: 'vehicle',
}

class MPPIOptimizer(Node):
    def __init__(self):
        super().__init__('mppi_optimizer')

        # 1. 테스트할 파라미터 범위 설정 (Grid Search)
        self.q_v_list = [1.3, 1.5, 1.7]
        self.q_dist_list = [0.0]
        self.q_du_list = [0.2, 0.5, 0.8]
        self.q_steer_list = [3.0, 5.0, 7.0 ]
        self.q_lat_g_list = [200.0, 250.0, 300.0]
        self.q_collision_list = [200.0, 150.0]
        self.q_progress_list = [10.0, 13.0, 16.0]
        self.q_escape_vel_list = [5.0, 6.5, 8.0]
        # 그립 한계(friction-ellipse) 소프트 코스트의 횡/종 방향 임계값도 탐색 대상에 포함
        # (params.yaml 기본값: lat_g_soft_limit=7.5, longitudinal_accel_soft_limit=4.0 근방)
        self.lat_g_soft_limit_list = [7.0, 7.5, 8.0]
        self.longitudinal_accel_soft_limit_list = [3.5, 4.0, 4.5]

        self.param_combinations = list(itertools.product(
            self.q_v_list, self.q_dist_list, self.q_du_list,
            self.q_steer_list, self.q_lat_g_list, self.q_collision_list, self.q_progress_list, self.q_escape_vel_list,
            self.lat_g_soft_limit_list, self.longitudinal_accel_soft_limit_list,
        ))
        self.get_logger().info(f"Total Combinations to run: {len(self.param_combinations)}")

        # 완주 기준: 10바퀴 이상 코스이탈/충돌 없이 주행
        self.required_laps = 10
        # 10바퀴 기준 타임아웃 (기존 3바퀴=60s에서 새 타이어 모델 기준으로 넉넉히 연장)
        self.run_timeout_sec = 300.0

        # 원본 yaml 경로 캐싱
        self.base_yaml_path = os.path.join(
            get_package_share_directory("smppi_cuda_controller"),
            "config",
            "params.yaml"
        )

        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.collision_sub = self.create_subscription(Bool, '/collision0', self.collision_callback, 10)
        # 벽/차량 충돌 원인 구분용 (ground-truth 분류, simulator.cpp classifyCollision 참고)
        self.collision_event_sub = self.create_subscription(
            CollisionEvent, '/collision_event0', self.collision_event_callback, 10)

        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        # 실제 시뮬레이터가 구독하는 토픽(/drive)으로 정지 명령을 보내야 한다.
        # (기존에는 /ackermann_cmd0로 보내고 있어 시뮬레이터에 명령이 전달되지 않는 버그가 있었음)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.results = []
        self.current_run = 0

        self.car_x = 0.0
        self.car_y = 0.0
        self.car_v = 0.0
        self.start_time = 0.0
        self.is_running = False
        self.has_crashed = False
        self.crash_source = CollisionEvent.NONE
        self.max_distance = 0.0
        self.mppi_process = None
        self.mppi_log_file = None

        # per-run lap counting and departure-wait flag
        self.lap_count = 0
        self.awaiting_departure = False
        self.crash_ignore_deadline = 0.0

        # 시작 포즈 및 완료 판정 기준
        self.start_x = 0.0
        self.start_y = 0.2
        self.start_yaw = 0.0
        self.min_lap_distance = 1.0

        self.reset_pending = False
        self.reset_deadline = 0.0

        # --- 결과 CSV: 매 실행마다 즉시 append(체크포인트) + 재시작 지원 ---
        os.makedirs("result", exist_ok=True)
        self.csv_path = "result/mppi_optimization_results.csv"
        self.current_run = self._resume_from_existing_csv()

        self.timer = self.create_timer(1.0, self.optimization_loop)

    def _resume_from_existing_csv(self) -> int:
        """기존 결과 CSV가 있으면 이미 완료된 조합 수를 세어 그 지점부터 이어서 시작한다."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
            return 0

        with open(self.csv_path, 'r', newline='') as f:
            completed = sum(1 for _ in csv.DictReader(f))

        self.get_logger().info(f"Resuming: {completed} runs already completed, continuing from run {completed + 1}")
        return completed

    def _append_result(self, row: dict):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerow(row)

    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        self.car_v = msg.twist.twist.linear.x

    def collision_callback(self, msg):
        # Ignore collision messages during reset/grace periods or when controller not running
        if time.time() < self.crash_ignore_deadline:
            return
        if not self.is_running:
            return
        if msg.data == True:
            self.has_crashed = True
            self.get_logger().warn("Simulator reported a CRASH!")

    def collision_event_callback(self, msg):
        # Ignore during reset/grace periods or when controller not running (동일한 정책, collision_callback과 별개 신호)
        if time.time() < self.crash_ignore_deadline:
            return
        if not self.is_running:
            return
        if msg.source != CollisionEvent.NONE:
            self.crash_source = msg.source

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
        self.crash_source = CollisionEvent.NONE
        # ignore collision messages briefly while simulator settles
        self.crash_ignore_deadline = time.time() + 1.0
        self.max_distance = 0.0
        self.reset_pending = True
        self.reset_deadline = time.time() + 1.0

    def start_mppi_node(self, q_v, q_dist, q_du, q_steer, q_lat_g, q_collision, q_progress, q_escape_vel,
                         lat_g_soft_limit, longitudinal_accel_soft_limit):
        """🚨 ros2 run으로 제어기만 단독 실행. 베이스 yaml 위에 최적화 변수만 덮어씌움"""
        cmd = [
            "ros2", "run", "smppi_cuda_controller", "smppi_node",
            "--ros-args",
            "--params-file", self.base_yaml_path,  # 기본 차량 세팅(D_f, mass 등)은 여기서 로드
            "-p", f"q_v:={q_v}",                   # 아래 변수들만 실시간 덮어쓰기
            "-p", f"q_dist:={q_dist}",
            "-p", f"q_du:={q_du}",
            "-p", f"q_steer:={q_steer}",
            "-p", f"q_lat_g:={q_lat_g}",
            "-p", f"q_collision:={q_collision}",
            "-p", f"q_progress:={q_progress}",
            "-p", f"q_escape_vel:={q_escape_vel}",
            "-p", f"lat_g_soft_limit:={lat_g_soft_limit}",
            "-p", f"longitudinal_accel_soft_limit:={longitudinal_accel_soft_limit}",
            "-p", "use_mcl_pose:=False"            # 시뮬레이터 모드 강제
        ]

        self.get_logger().info(
            f"Run {self.current_run + 1}/{len(self.param_combinations)}: q_v={q_v}, lat_g={q_lat_g}, "
            f"col={q_collision}, progress={q_progress}, escape_vel={q_escape_vel}, "
            f"lat_g_soft={lat_g_soft_limit}, long_accel_soft={longitudinal_accel_soft_limit}")

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
        self.lap_count = 0
        self.awaiting_departure = False
        self.max_distance = 0.0
        # clear crash flag and set short grace period to ignore stale collision pub
        self.has_crashed = False
        self.crash_source = CollisionEvent.NONE
        self.crash_ignore_deadline = time.time() + 1.0
        self.is_running = True

    def stop_mppi_node(self):
        """제어기 노드 강제 종료"""
        # publish explicit stop command to halt vehicle immediately
        try:
            stop_msg = AckermannDriveStamped()
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(stop_msg)
        except Exception:
            pass
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
            params = self.param_combinations[self.current_run]
            self.start_mppi_node(*params)

        else:
            elapsed_time = time.time() - self.start_time
            distance_from_start = math.hypot(self.car_x - self.start_x, self.car_y - self.start_y)
            self.max_distance = max(self.max_distance, distance_from_start)

            is_crashed = self.has_crashed
            # clear awaiting_departure once vehicle leaves start area
            if self.awaiting_departure and distance_from_start > 2.0:
                self.awaiting_departure = False

            is_lap_condition = (elapsed_time > 10.0) and (distance_from_start < 2.0) and (self.max_distance > self.min_lap_distance)
            is_timeout = elapsed_time > self.run_timeout_sec

            if is_crashed:
                # immediate stop on crash
                self.stop_mppi_node()
                self._finish_run('Crashed', 999.0)

            elif is_lap_condition and not self.awaiting_departure:
                # completed one lap
                self.lap_count += 1
                self.awaiting_departure = True
                self.max_distance = 0.0
                self.get_logger().info(f"Lap {self.lap_count}/{self.required_laps} completed for run {self.current_run + 1}")

                # if reached required laps, finish run
                if self.lap_count >= self.required_laps:
                    self.stop_mppi_node()
                    self._finish_run('Finished', elapsed_time)

            elif is_timeout:
                # timeout for the run
                self.stop_mppi_node()
                self._finish_run('Timeout', 999.0)

    def _finish_run(self, status: str, lap_time: float):
        q_v, q_dist, q_du, q_steer, q_lat_g, q_col, q_progress, q_escape_vel, \
            lat_g_soft_limit, longitudinal_accel_soft_limit = \
            self.param_combinations[self.current_run]
        collision_source = _COLLISION_SOURCE_NAMES.get(self.crash_source, 'none') if status == 'Crashed' else ''
        row = {
            'q_v': q_v, 'q_dist': q_dist, 'q_du': q_du, 'q_steer': q_steer,
            'q_lat_g': q_lat_g, 'q_collision': q_col, 'q_progress': q_progress, 'q_escape_vel': q_escape_vel,
            'lat_g_soft_limit': lat_g_soft_limit,
            'longitudinal_accel_soft_limit': longitudinal_accel_soft_limit,
            'status': status, 'lap_time': lap_time, 'max_distance': self.max_distance,
            'collision_source': collision_source,
        }
        self._append_result(row)
        self.get_logger().info(
            f"Ended: {status}, laps={self.lap_count}, Time: {lap_time:.2f}s, "
            f"Dist: {self.max_distance:.2f}m, CollisionSource: {collision_source or 'n/a'}")
        self.current_run += 1


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
