import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import cv2
import datetime
import time



# 카메라 영상을 받아올 객체 선언 및 설정(영상 소스, 해상도 설정)
capture = cv2.VideoCapture(6)
if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit(1)


capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)




fourcc = cv2.VideoWriter_fourcc(*'XVID')    # 영상을 기록할 코덱 설정
is_record = False                           # 녹화상태는 처음엔 거짓으로 설정
is_drawing_complete = False  # 전역 변수로 선언




class LidarDistanceMeasurer(Node):
    def __init__(self):
        super().__init__('lidar_distance_measurer')

        # Initialize a subscriber to the LaserScan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        # Publisher to send commands to the robot's wheels
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Flag to control the flow of rotation and measurement
        self.rotation_complete = False
        self.distance_measurement_complete= False

        # Initialize distances list
        self.distances = []

        self.get_logger().info("Lidar distance measurer node has been started.")

    def rotate_robot(self):
        """Rotates the robot 90 degrees to the right at a set angular velocity."""
        twist = Twist()
        twist.angular.z = -0.3  # Negative for right rotation, adjust speed as needed
        self.vel_pub.publish(twist)

        # Schedule to stop rotation after enough time has passed to rotate 90 degrees
        rotation_duration = (math.pi / 2 / abs(twist.angular.z))*1.1  # Time = angle / angular velocity
        self.timer = self.create_timer(rotation_duration, self.stop_rotation)

    def stop_rotation(self):
        """Stops the robot's rotation."""
        twist = Twist()  # Zero velocity
        self.vel_pub.publish(twist)
        time.sleep(2)
        self.rotation_complete = True
        self.timer.cancel()  # Stop the timer to prevent further calls
        self.get_logger().info("Stopped rotating the robot.")


    def scan_callback(self, msg):
        # if not self.rotation_complete:
        #     return
        if self.distance_measurement_complete:
            return

        # Define the range of angles to check (from -10 degrees to 10 degrees)
        start_angle = -10.0
        end_angle = 10.0
        angle_increment = 1.0  # assuming each LiDAR reading corresponds to 1 degree

        min_distance = float('inf')
        valid_distance_found = False

        # Check distances within the -10 to 10 degrees range
        for angle in range(int(start_angle), int(end_angle) + 1):
            index = int((angle / 360.0) * len(msg.ranges)) % len(msg.ranges)
            distance = msg.ranges[index]

            if distance != float('inf') and distance != 0.0:
                valid_distance_found = True
                if distance < min_distance:
                    min_distance = distance

        if not valid_distance_found:
            self.get_logger().info("No obstacle detected within 10 degrees range of the front side of the robot.")
        else:
            distance = min_distance
            if distance > 1.0:
                self.get_logger().info(f"Object detected {distance:.3f} meters within 10 degrees range of the front side of the robot, but too far.")
            else:
                self.get_logger().info(f"Object detected {distance:.3f} meters within 10 degrees range of the front side of the robot.")
                self.distances.append(distance)
                if len(self.distances) >= 10:
                    self.get_logger().info(f"Distance measurement complete")
                    self.distance_measurement_complete = True
                    self.get_logger().info(f"Shutting down LIDAR distance measurement")
                    self.scan_sub.destroy()
                    # self.rotation_complete=True
                    # Rotate the robot 90 degrees to the left
                    self.rotate_robot()



class CircleDrawer(Node):
    def __init__(self, radius):
        super().__init__('turtlebot3_circle')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.radius = radius
        self.angular_speed = 0.1  # 시계 방향 회전을 위해 각속도 음수 설정
        self.linear_speed = abs(self.angular_speed) * self.radius  # 선형 속도 설정
        self.target_angle = 2.0 * math.pi  # 360도 회전 목표
        self.current_angle = 0

        # 타이머 설정
        self.timer = self.create_timer(0.01, self.draw_circle)  # 0.1초 간격으로 draw_circle 호출

    def draw_circle(self):
        if self.current_angle < self.target_angle:
            speed = Twist()
            speed.linear.x = float(self.linear_speed)  # 명확하게 float 타입을 지정
            speed.angular.z = float(self.angular_speed)  # 명확하게 float 타입을 지정
            self.publisher.publish(speed)

            # 각도 업데이트
            self.current_angle += abs(self.angular_speed) * 0.1395  # 타이머 주기에 맞춰 각도 업데이트
        else:
            # 원 그리기 완료 후 정지 메시지 전송하고 타이머 정지
            speed = Twist()
            speed.linear.x = float(0)  # 명확하게 float 타입을 지정
            speed.angular.z = float(0)  # 명확하게 float 타입을 지정
            self.publisher.publish(speed)
            self.timer.cancel()  # 타이머 정지
            global is_drawing_complete
            is_drawing_complete = True


def main(args=None):
    rclpy.init(args=args)
    lidar_node = LidarDistanceMeasurer()
    circle_drawer = None  # 초기에 None으로 설정
    video = None
    global is_record

    try:
        # rclpy.spin_once(node)  # Keep the node active to listen to incoming data
        # node.destroy_node()
        while rclpy.ok():
            ret, frame = capture.read()
            if is_record:
                display_frame = frame.copy()  # 녹화용 프레임과 화면 표시용 프레임을 분리
                cv2.circle(img=display_frame, center=(620, 15), radius=5, color=(0,0,255), thickness=-1)
                cv2.imshow("Camera Feed", display_frame)
            else:
                cv2.imshow("Camera Feed", frame)

            # 현재시각을 불러와 문자열로저장
            if is_record and video:
                video.write(frame)

            key = cv2.waitKey(1) & 0xFF

            if not is_record and lidar_node.rotation_complete:#circle_drawer
                #radius = float(input("원의 반지름을 센티미터 단위로 입력하세요: "))
                avg_distance = lidar_node.distances[9]
                radius = avg_distance * 100  # meters to centimetersi
                print("Drawing a circle which has "+str(radius+4)+ "cm of radius")
                circle_drawer = CircleDrawer((radius+4) / 100)
                is_record = True  # 원 그리기 시작과 동시에 녹화 시작
                #rclpy.spin_once(circle_drawer, timeout_sec=0.1)  # 반복적으로 spin_once를 호출하도록변경

            if is_record and not video:
                now = datetime.datetime.now()
                nowDatetime_path = now.strftime('%m-%d_%H_%M_cap')
                video = cv2.VideoWriter(nowDatetime_path + ".avi", fourcc, 10, (640,480))


            if is_drawing_complete and is_record:
                if video:
                    video.release()
                    video = None
                if circle_drawer:
                    circle_drawer.destroy_node()
                    circle_drawer = None  # 노드 파괴 후 None으로 설정
                print("녹화 완료, 프로그램 종료합니다.")
                break
            # 원 그리기와 녹화를 처리하기 위해 주기적으로 spin
            if circle_drawer:
                rclpy.spin_once(circle_drawer, timeout_sec=0.1)
            else:
                rclpy.spin_once(lidar_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if video:
            video.release()
        capture.release()
        cv2.destroyAllWindows()
        if circle_drawer:
            circle_drawer.destroy_node()
        lidar_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

