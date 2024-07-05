# detect_move.py
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python3 detect_capstone.py --weights NeRFBot_capstone.pt --source 0   0:logitech RGB, 4: RealSense depth, 6: RealSense RGB
"""

import argparse
import csv
import os
import platform
import sys
from models.experimental import attempt_load  ###24.05.23 for load weights pt files
from pathlib import Path, PureWindowsPath ### 24.05.23 for path
#import os ### for robot
import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile ###
import threading ### 04.30
import subprocess
import time
import torch
import select
import termios
import tty

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode




######################### robot move ############################### 24.04.11
# 상수 설정
BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.05

TURTLEBOT3_MODEL = os.getenv('TURTLEBOT3_MODEL', 'burger')

keyboard_control_enabled = True ###
target_label_input_mode = False

msg = """\n
Control Your TurtleBot3!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity (Burger : ~ 0.22)
a/d : increase/decrease angular velocity (Burger : ~ 2.84)
space key, s : force stop

CTRL-C to quit\n
"""

# ROS2 노드 및 퍼블리셔 초기화
rclpy.init()
qos = QoSProfile(depth=10)
node = rclpy.create_node('move')
pub = node.create_publisher(Twist, 'cmd_vel', qos)

# 속도 제한 함수
def constrain(input_vel, low_bound, high_bound):
    return max(low_bound, min(high_bound, input_vel))

# 선형 속도 설정
def set_linear_velocity(velocity):
    velocity = constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    twist = Twist()
    twist.linear.x = velocity
    pub.publish(twist)
    node.get_logger().info(f"Setting linear velocity: {velocity}")

# 각속도 설정
def set_angular_velocity(velocity):
    velocity = constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    twist = Twist()
    twist.angular.z = velocity
    pub.publish(twist)
    node.get_logger().info(f"Setting angular velocity: {velocity}")
def stop_movement():
    twist = Twist()
    pub.publish(twist)
    node.get_logger().info("Stopping movement")

def release_camera_resources():
    """카메라 자원을 반환하고 OpenCV 창을 닫습니다."""
    if 'dataset' in globals():
        if hasattr(dataset, 'cap'):
            for cap in dataset.cap:
                if cap:
                    print("release cap")
                    cap.release()
        elif hasattr(dataset, 'vid_cap'):
            for vid_cap in dataset.vid_cap:
                if vid_cap:
                    print("release vid_cap")
                    vid_cap.release()
    if 'capture' in globals():
        capture.release()
    cv2.destroyAllWindows()
    print("카메라 자원과 OpenCV 창이 모두 반환되었습니다.")


def run_orbit_record():
    time.sleep(3)

    orbit_record_path = Path.home() / "orbit_record_complete.py"
    subprocess.run([sys.executable, str(orbit_record_path)])
    sys.exit()


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity, target_angular_velocity))

def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input
    return output

def check_linear_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)


def check_angular_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)


def keyboard_control():
    global keyboard_control_enabled, target_label_input_mode, target_label
    settings = termios.tcgetattr(sys.stdin)
    try:
        target_linear_velocity = 0.0
        target_angular_velocity = 0.0

        while True:

            key = get_key(settings)

            if target_label_input_mode:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)  # cooked 모드로 전환
                print("Input target label: ", end='', flush=True)
                target_label = sys.stdin.readline().strip()  # stdin에서 입력 읽기
                target_label_input_mode = False
                termios.tcsetattr(sys.stdin, termios.TCSANOW,settings)# tty.setraw(sys.stdin.fileno()))  # raw 모드로 복귀
                print(f"\nTarget label set to: {target_label}")
                keyboard_control_enabled = True  # 키보드 조작 활성화
                continue

            if not keyboard_control_enabled and label == target_label:
                print("\n=========================================\nkeyboard operation disabled\n========================================\n")
                termios.tcsetattr(sys.stdin, termios.TCSANOW, tty.setraw(sys.stdin.fileno()))
                time.sleep(1)  # 1초 동안 대기
                continue



            if key == 'o':
                target_label_input_mode = True
                target_label = ""
                #keyboard_control_enabled = False  # target_label 입력 동안 키보드 조작 차단
                continue
           
            if key == 'w':
                target_linear_velocity = check_linear_limit_velocity(target_linear_velocity + LIN_VEL_STEP_SIZE)
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == 'x':
                target_linear_velocity = check_linear_limit_velocity(target_linear_velocity - LIN_VEL_STEP_SIZE)
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == 'a':
                target_angular_velocity = check_angular_limit_velocity(target_angular_velocity + ANG_VEL_STEP_SIZE)
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == 'd':
                target_angular_velocity = check_angular_limit_velocity(target_angular_velocity - ANG_VEL_STEP_SIZE)
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == ' ' or key == 's':
                target_linear_velocity = 0.0
                target_angular_velocity = 0.0
                print_vels(target_linear_velocity, target_angular_velocity)
            else:
                if key == '\x03':
                    break

            twist = Twist()
            twist.linear.x = target_linear_velocity
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = target_angular_velocity
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        stop_movement()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)









@smart_inference_mode()
def run(
    #weights=ROOT / "yolov5s.pt",  # model path or triton URL
    weights= "/home/yolov5/capstone_NeRFBot.pt",
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    global dataset, target_label, label ### 06.03
    label = None
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "tcp://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    target_label = None
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:

                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    xmin, ymin, xmax, ymax = map(int, xyxy)

                    box_width = xmax- xmin
                    box_height = ymax - ymin
                    #frame_center
                    frame_center_x = im0.shape[1] / 2
                    frame_center_y = im0.shape[0] / 2
                    box_center_x = (xmin + xmax) / 2
                    box_center_y = (ymin + ymax) / 2

                    dist_x = box_center_x - frame_center_x
                    dist_y = box_center_y - frame_center_y

                    #bbox print
                    if label == target_label:
                        print(f"\nClass ID: {label}, BBox: X1Y1({xmin}, {ymax}), X4Y4({xmax}, {ymin}), Width: {box_width}, Height: {box_height}")

                    if label == target_label:
                        xmin, ymin, xmax, ymax = map(int, xyxy)
                        box_width = xmax - xmin
                        box_height = ymax - ymin

                        global keyboard_control_enabled   ###
                        keyboard_control_enabled = False  ###

                        if abs(dist_x) > 50:
                            if dist_x >0:
                                print("TurtleBot : move the camera toward right")
                                set_angular_velocity(-0.02)  # 오른쪽으로 회전
                            else :
                                print("TurtleBot : move the camera toward left")
                                set_angular_velocity(0.02)  # 왼쪽으로 회전
                        #if abs(dist_y) >50:   # y축은 고정이므로 필요 x
                            #if dist_y > 0:
                                #print("move the camera upward")
                                #print("move the camera downward")
                        if abs(dist_x) <=50:#and abs(dist_y) <= 50:
                            #print("the target is located center of the camera")
                                                    #640 * 480
                            if box_width < 200 and box_height < 200:
                                print("TurtleBot : Get closer to the target")
                                set_linear_velocity(0.03)  # 빨리 앞으로

                            elif box_width < 250 and box_height < 250:
                                print("TurtleBot : Get closer to the target")
                                set_linear_velocity(0.02)  # 앞으로

                            elif box_width < 300 and box_height < 300:
                                print("TurtleBot : Get closer to the target")
                                set_linear_velocity(0.01)  # 천천히 앞으로

                            elif box_width >= 450 or box_height >= 400:
                                print("TurtleBot : Get further from the target")
                                set_linear_velocity(-0.01)  # 천천히 뒤로

                            elif box_width >= 550 or box_height >= 480:
                                print("TurtleBot : Get further from the target")
                                set_linear_velocity(-0.02)  # 뒤로

                            if (box_width < 450 and box_width >300) or (box_height > 300 and box_height < 480):
                                if abs(dist_x) > 20:
                                    if dist_x > 0:
                                        print("TurtleBot : move the camera toward right")
                                        set_angular_velocity(-0.01)  # 오른쪽으로 회전
                                    else :
                                        print("TurtleBot : move the camera toward left")
                                        set_angular_velocity(0.01)  # 왼쪽으로 회전
                                if abs(dist_x) <=20:#and abs(dist_y) <= 50:
                                    print("TurtleBot : The Target is ready for recording.")
                                    stop_movement()  # 정지
                                    release_camera_resources()
                                    node.destroy_node()
                                    rclpy.shutdown()
                                    run_orbit_record()  ####24.05.02


                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)


            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        # if cv2.waitKey(1) & 0xFF == ord('o'):
        #     target_label = input("input target_label :  ")
        #     print(f"target_label name : {target_label}")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    print(msg)
    # 키보드 입력 처리를 위한 스레드 시작
    keyboard_thread = threading.Thread(target=keyboard_control)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    run(**vars(opt))


if __name__ == "__main__":
        try:
            opt = parse_opt()
            main(opt)
            # stop_movement()  # 종료 시 정지
            # node.destroy_node()
            # rclpy.shutdown()

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. program down")
            stop_movement()
            node.destroy_node()
            rclpy.shutdown()
