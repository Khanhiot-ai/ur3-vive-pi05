#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import serial
import struct
import logging
import argparse
import numpy as np
from serial.tools import list_ports
from pika.sense import Sense as PikaSense
from pika.gripper import Gripper as PikaGripper
from xarm.wrapper import XArmAPI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('robot_teleop')


def get_serial_ports(vidpid='1a86:7522'):
    """
    搜索所有指定vidpid的串口
    vidpid: 指定设备的VID:PID字符串, 默认值为'1a86:7522'
    返回找到的所有符合的串口号列表
    """
    ports = list_ports.comports()
    pika_ports = []
    for port in ports:
        if port.vid is not None and port.pid is not None:
            if '{:04x}:{:04x}'.format(port.vid, port.pid) == vidpid:
                pika_ports.append(port.device)
            # else:
            #     print('pidvid:', '{:04x}:{:04x}'.format(port.vid, port.pid))
    return pika_ports

def check_pika_device(port):
    """
    检测串口对应的Pika设备类型
    返回值:
        -1: 无法打开串口
        0: 不是Pika设备
        1: Pika Sense设备
        2: Pika Gripper设备
    """
    try:
        ser = serial.Serial(
            port=port,
            baudrate=460800,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0
        )
        time.sleep(0.5)  # 等待串口稳定
        data = b''
        expired_time = time.monotonic() + 1.0  # 最多等待1秒
        while time.monotonic() < expired_time:
            if ser.in_waiting > 0:
                data += ser.read(ser.in_waiting)
                if len(data) > 200:  # 足够的数据来判断
                    break
            time.sleep(0.05)
        ser.close()
        data_str = data.decode('utf-8', errors='ignore')
        if '"Command"' in data_str or '"AS5047"' in data_str or '"IMU"' in data_str:
            logger.info('✓ 检测到 Pika Sense 设备: {}'.format(port))
            return 1
        elif '"motor"' in data_str or '"motorstatus"' in data_str:
            logger.info('✓ 检测到 Pika Gripper 设备: {}'.format(port))
            return 2
        else:
            logger.info('✗ 未检测到 Pika 设备: {}, 数据长度: {}'.format(port, len(data)))
            return 0
    except:
        pass
    return -1


class Transformations:
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        将四元数转换为旋转矩阵
        
        注: 四元素顺序为xyzw
        """
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            raise ValueError('零四元数无法归一化')

        x, y, z, w = q / norm  # 归一化
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = np.array([
            [1 - 2 * (yy + zz),     2 * (xy - wz),      2 * (xz + wy)],
            [    2 * (xy + wz), 1 - 2 * (xx + zz),      2 * (yz - wx)],
            [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])
        return R

    @staticmethod
    def rpy_to_rotation_matrix(roll, pitch, yaw):
        """RPY角到旋转矩阵的转换"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cp*cy,  -cr*sy + sr*sp*cy,    sr*sy + cr*sp*cy],
            [cp*sy,   cr*cy + sr*sp*sy,   -sr*cy + cr*sp*sy],
            [ -sp,        sr*cp,               cr*cp],
        ])

        return R
    
    @staticmethod
    def rotation_matrix_to_rpy(R, yaw_zero=True):
        """
        旋转矩阵到RPY角的转换
        
        yaw_zero: 万向节锁情况下, True就把yaw置0, False就把roll置0
        """
        epsilon = 1e-6
        if abs(R[2, 0]) > 1 - epsilon: # 万向节锁(pitch=±90°)
            pitch = np.arcsin(-R[2, 0])
            roll_yaw = np.arctan2(-R[0, 1], R[1, 1])
            if yaw_zero:
                # 保留roll, 把yaw置0
                roll, yaw = roll_yaw, 0
            else:
                # 保留yaw, 把roll置0
                roll, yaw = 0, roll_yaw
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        return roll, pitch, yaw


class UFRobotTeleop(object):
    def __init__(self, args, pika_to_robot_eef=None, **kwargs):

        self.pika_to_robot_eef = [0, 0, 0, math.pi, 0, 0] if pika_to_robot_eef is None else pika_to_robot_eef

        if isinstance(args, argparse.Namespace):
            robot_ip = args.robot_ip
            self.robot_mode = args.robot_mode
            self.robot_speed = args.robot_speed
            self.robot_acc = args.robot_acc
            self.gripper_type = args.gripper_type
            self.gripper_speed = args.gripper_speed
            self.gripper_force = args.gripper_force
            self.use_modbus = args.use_modbus
            # self.fixed_roll = args.fixed_roll
            # self.fixed_pitch = args.fixed_pitch
            # self.fixed_yaw = args.fixed_yaw
        else:
            robot_ip = args
            self.robot_mode = kwargs.get('robot_mode', 7)
            self.robot_speed = 1000
            self.robot_acc = 5000
            self.gripper_type = kwargs.get('gripper_type', 0)
            self.gripper_speed = 5000 if self.gripper_type == 1 else 225 if self.gripper_type == 2 else 4500 if self.gripper_type == 3 else 0
            self.gripper_force = 0 if self.gripper_type == 1 else 50 if self.gripper_type == 2 else 100 if self.gripper_type == 3 else 0
            self.use_modbus = False
            # self.fixed_roll = 0
            # self.fixed_pitch = 0
            # self.fixed_yaw = 0

        pika_ports = get_serial_ports()
        if not pika_ports:
            logger.error('未找到Pika设备, 请检查连接')
            exit(1)

        pika_sense_port = None
        pika_gripper_port = None
        for port in pika_ports:
            if pika_sense_port is None or (self.gripper_type == 10 and pika_gripper_port is None):
                device_type = check_pika_device(port)
                if device_type == 1:
                    pika_sense_port = port
                    if self.gripper_type != 10:
                        break
                elif device_type == 2:
                    pika_gripper_port = port

        if pika_sense_port is None:
            logger.error('未找到Pika Sense设备, 请检查连接')
            exit(1)

        if self.gripper_type == 10 and pika_gripper_port is None:
            logger.error('未找到Pika Gripper设备, 请检查连接')
            exit(1)

        print('Pika Sense Port:', pika_sense_port)
        print('Pika Gripper Port:', pika_gripper_port)

        self.arm = XArmAPI(robot_ip, is_radian=True)
        
        # logger.info('开始获取WM0设备的位姿数据...')

        # 初始化Sense对象
        self.pika_sense = PikaSense(port=pika_sense_port)
        # 连接设备
        if not self.pika_sense.connect():
            logger.error('连接Pika Sense设备失败')
            exit(1)
        logger.info('Pika Sense设备连接成功')
        
        # 配置Vive Tracker（可选）
        # sense.set_vive_tracker_config(config_path='path/to/config', lh_config='lighthouse_config')

        tracker = self.pika_sense.get_vive_tracker()
        if not tracker:
            logger.error('Vive Tracker初始化失败')
            self.pika_sense.disconnect()
            exit(1)
        logger.info('Vive Tracker初始化成功')
        time.sleep(2)

        devices = self.pika_sense.get_tracker_devices()
        if not devices:
            logger.error('未检测到Vive Tracker设备')
            self.pika_sense.disconnect()
            exit(1)
        logger.info('检测到Vive Tracker设备: {}'.format(devices))

        self.target_device = None
        for device in devices:
            if device.startswith('WM'):
                self.target_device = device
                break
        else:
            self.target_device = devices[0]
        logger.info('开始跟踪设备: {}\n'.format(self.target_device))

        if self.gripper_type == 10:
            self.pika_gripper = PikaGripper(port=pika_gripper_port)
            # 连接设备
            if not self.pika_gripper.connect():
                logger.error('连接Pika Gripper设备失败')
                self.pika_sense.disconnect()
                exit(1)
            logger.info('Pika Gripper设备连接成功')
        else:
            self.pika_gripper = None

    @staticmethod
    def xyzq_to_rotation_matrix(x, y, z, q):
        T = np.eye(4)
        T[:3, :3] = Transformations.quaternion_to_rotation_matrix(q)
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def xyzrpy_to_rotation_matrix(x, y, z, roll, pitch, yaw):
        """构造4x4齐次变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = Transformations.rpy_to_rotation_matrix(roll, pitch, yaw)
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def rotation_matrix_to_xyzrpy(rotation_matrix):
        """从4x4齐次变换矩阵到xyzrpy的转换"""
        x, y, z = rotation_matrix[0, 3], rotation_matrix[1, 3], rotation_matrix[2, 3]
        roll, pitch, yaw = Transformations.rotation_matrix_to_rpy(rotation_matrix)
        return [x, y, z, roll, pitch, yaw]

    def pika_pose_to_robot_matrix(self, x, y, z, q, pika_to_robot_matrix):
        # pika位置对应的变换矩阵
        pika_matrix = self.xyzq_to_rotation_matrix(x, y, z, q)
        # pika位置转换到机械臂坐标系后对应的变换矩阵
        robot_matrix = np.dot(pika_matrix, pika_to_robot_matrix)
        return robot_matrix
    
    def pika_robot_matrix_to_robot_pose(self, pika_begin_robot_matrix, pika_end_robot_matrix, robot_base_matrix):
        # 机械臂目标位置对应的变换矩阵
        robot_martix = np.dot(robot_base_matrix, np.dot(np.linalg.inv(pika_begin_robot_matrix), pika_end_robot_matrix))
        return self.rotation_matrix_to_xyzrpy(robot_martix)
    
    def robot_init(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(self.robot_mode)
        self.arm.set_state(0)

        if self.gripper_type == 1 or self.gripper_type == 2:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)
            self.arm.set_gripper_speed(self.gripper_speed)
        elif self.gripper_type == 3:
            self.arm.set_bio_gripper_enable(True)
        elif self.gripper_type == 10:
            self.pika_gripper.enable()

        # if self.gripper_type == 1:
        #     # gripper enable
        #     datas = [0x08, 0x10, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x01]
        #     self.arm.getset_tgpio_modbus_data(datas=datas)
        #     # gripper speed
        #     datas = [0x08, 0x10, 0x03, 0x03, 0x00, 0x01, 0x02, (self.gripper_speed >> 8) & 0xFF, self.gripper_speed & 0xFF]
        #     self.arm.getset_tgpio_modbus_data(datas=datas)
    
    def set_robot_position(self, pose):
        # logger.info('[运动]: {}, {}, {}, {}, {}, {}'.format(*pose))
        if self.robot_mode == 7:
            # mode 7
            return self.arm.set_position(*pose, radius=0, speed=self.robot_speed, mvacc=self.robot_acc)
        else:
            # mode 1
            return self.arm.set_servo_cartesian(pose, speed=self.robot_speed, mvacc=self.robot_acc)

    def set_gripper_position(self, distance):
        if self.gripper_type == 1:
            pos = int(0 + (distance / 100) * (850 - 0))
            if self.use_modbus:
                datas = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04, (pos >> 24) & 0xFF, (pos >> 16) & 0xFF, (pos >> 8) & 0xFF, pos & 0xFF]
                code, _ = self.arm.getset_tgpio_modbus_data(datas=datas)
                return code
            else:
                return self.arm.set_gripper_position(pos, speed=self.gripper_speed, wait=False, wait_motion=False)
        elif self.gripper_type == 2:
            pos = int(0 + (distance / 100) * (84 - 0))
            pos = min(max(0, pos), 84)
            if self.use_modbus:
                pos = int((math.degrees(math.asin((pos - 16) / 110)) + 8.33) * 18.28)
                speed = int(((self.gripper_speed * 60) / 9.88235 + 140) / 0.4)
                datas = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
                datas.extend(list(struct.pack('>h', speed))) # speed // 256 % 256, speed % 256
                datas.extend(list(struct.pack('>h', self.gripper_force))) # force // 256 % 256, force % 256
                datas.extend(list(struct.pack('>i', pos)))
                code, _ = self.arm.getset_tgpio_modbus_data(datas=datas)
                return code
            else:
                return self.arm.set_gripper_g2_position(pos, speed=self.gripper_speed, force=self.gripper_force, wait=False, wait_motion=False)
        elif self.gripper_type == 3:
            pos = 71 + (distance / 100) * (150 - 71)
            pos = min(max(71, pos), 150)
            if self.use_modbus:
                pos_pluse = int(pos * 3.7342 - 265.13)
                datas = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
                datas.extend(list(struct.pack('>h', self.gripper_speed))) # speed // 256 % 256, speed % 256
                datas.extend(list(struct.pack('>h', self.gripper_force))) # force // 256 % 256, force % 256
                datas.extend(list(struct.pack('>i', pos_pluse)))
                code, _ = self.arm.getset_tgpio_modbus_data(datas=datas)
                return code
            else:
                return self.arm.set_bio_gripper_g2_position(pos, speed=self.gripper_speed, force=self.gripper_force, wait=False, wait_motion=False)
        elif self.gripper_type == 10:
            # Pika Gripper: distance 从 Pika Sense 获取 (0-100mm)
            # Pika Gripper 范围是 0-100mm
            target_distance_mm = min(distance, 100)  # 限制最大值为100mm
            return self.pika_gripper.set_gripper_distance(target_distance_mm)

    def run(self):
        init_state = self.pika_sense.get_command_state()
        curr_state = init_state

        last_gripper_distance = 0

        ctrl_flag = False # 是否开启遥操作
        need_initial = False
        frequency = 100
        sleep_time = 1 / frequency

        self.robot_init()

        self.arm.set_linear_spd_limit_factor(2.0)

        # if self.gripper_type == 1 or self.gripper_type == 2:
        #     self.arm.set_gripper_enable(True)
        #     self.arm.set_gripper_mode(0)
        #     self.arm.set_gripper_speed(5000)
        # elif self.gripper_type == 3:
        #     self.arm.set_bio_gripper_enable(True)
        # elif self.gripper_type == 10:
        #     self.pika_gripper.enable()

        # pika坐标系到机械臂坐标系的变换关系对应的变换矩阵
        pika_to_robot_matrix = self.xyzrpy_to_rotation_matrix(*self.pika_to_robot_eef)
        # 机械臂初始位置对应的变换矩阵
        robot_base_matrix = None
        # pika初始位置转换到机械臂坐标系后对应的变换矩阵
        pika_begin_robot_matrix = None
        # pika目标位置转换到机械臂坐标系后对应的变换矩阵
        pika_end_robot_matrix = None

        robot_base_pose = None

        while True:
            time.sleep(sleep_time)

            state = self.pika_sense.get_command_state()
            if state != curr_state:
                curr_state = state
                if not ctrl_flag and curr_state != init_state:
                    ctrl_flag = True
                    need_initial = True
                    self.robot_init()
                    logger.info('开始遥操作')
                    time.sleep(1)
                elif ctrl_flag and curr_state == init_state:
                    ctrl_flag = False
                    logger.info('停止遥操作')
                    continue
            
            if ctrl_flag and (not self.arm.connected or self.arm.error_code != 0 or self.arm.state >= 4):
                logger.info('机械臂原因, 遥操作自动停止')
                init_state = state
                curr_state = state
                ctrl_flag = False
                continue
            
            if not ctrl_flag:
                continue

            if self.gripper_type > 0:
                distance  = self.pika_sense.get_gripper_distance()

                if abs(last_gripper_distance - distance) > 2:
                    last_gripper_distance = distance
                    self.set_gripper_position(distance)

            pose = self.pika_sense.get_pose(self.target_device)
            if not pose:
                continue
            x, y, z = pose.position[0] * 1000, pose.position[1] * 1000, pose.position[2] * 1000

            if need_initial:
                need_initial = False
                _, robot_pos = self.arm.get_position()
                robot_base_pose = robot_pos
                logger.info('[初始] 机械臂位置: {}'.format(robot_pos))

                # 机械臂初始位置对应的变换矩阵
                robot_base_matrix = self.xyzrpy_to_rotation_matrix(*robot_pos)

                # pika初始位置转换到机械臂坐标系后对应的变换矩阵
                pika_begin_robot_matrix = self.pika_pose_to_robot_matrix(x, y, z, pose.rotation, pika_to_robot_matrix)
                pika_end_robot_matrix = pika_begin_robot_matrix
            else:
                # pika目标位置转换到机械臂坐标系后对应的变换矩阵
                pika_end_robot_matrix = self.pika_pose_to_robot_matrix(x, y, z, pose.rotation, pika_to_robot_matrix)

            robot_target_pose = self.pika_robot_matrix_to_robot_pose(pika_begin_robot_matrix, pika_end_robot_matrix, robot_base_matrix)

            # if self.fixed_roll > 0 or self.fixed_pitch > 0 or self.fixed_yaw > 0:
            #     robot_target_pose[3] = math.radians(180) if self.fixed_roll == 2 else robot_base_pose[3] if self.fixed_roll == 1 else robot_target_pose[3]
            #     robot_target_pose[4] = math.radians(0) if self.fixed_pitch == 2 else robot_base_pose[4] if self.fixed_pitch == 1 else robot_target_pose[4]
            #     robot_target_pose[5] = math.radians(0) if self.fixed_yaw == 2 else robot_base_pose[5] if self.fixed_yaw == 1 else robot_target_pose[5]

            self.set_robot_position(robot_target_pose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_ip', type=str, help="robot ip address")
    parser.add_argument('robot_mode', type=int, nargs='?', default=7, help="robot mode (default: 7) [1: servo motion mode, 7: cartesian online trajectory planning mode]")
    parser.add_argument('--robot_mode', type=int, required=False, default=7, help="robot mode (default: 7) [1: servo motion mode, 7: cartesian online trajectory planning mode]")
    parser.add_argument('gripper_type', type=int, nargs='?', default=0, help="gripper type (default: 0) [0: no gripper, 1: xArm Gripper, 2: xArm Gripper G2, 3: BIO Gripper G2]")
    parser.add_argument('--gripper_type', type=int, required=False, default=0, help="gripper type (default: 0) [0: no gripper, 1: xArm Gripper, 2: xArm Gripper G2, 3: BIO Gripper G2]")

    parser.add_argument('--robot_speed', type=int, required=False, default=1000, help="robot speed (default: 1000)")
    parser.add_argument('--robot_acc', type=int, required=False, default=5000, help="robot acc (default: 5000)")
    parser.add_argument('--gripper_speed', type=int, required=False, help="gripper speed")
    parser.add_argument('--gripper_force', type=int, required=False, help="gripper force")
    parser.add_argument('--use_modbus', action='store_true', help="use modbus to control gripper or not")
    # parser.add_argument('--fixed_roll', type=float, required=False, default=0, help="fixed roll (default: 0, 0 means not fixed, 1 means use robot current roll, 2 means use 180 degrees)")
    # parser.add_argument('--fixed_pitch', type=float, required=False, default=0, help="fixed pitch (default: 0, 0 means not fixed, 1 means use robot current pitch, 2 means use 0 degrees)")
    # parser.add_argument('--fixed_yaw', type=float, required=False, default=0, help="fixed yaw (default: 0, 0 means not fixed, 1 means use robot current yaw, 2 means use 0 degrees)")

    args = parser.parse_args()
    args.robot_mode = 7 if args.robot_mode not in [1, 7] else args.robot_mode
    args.gripper_type = 0 if args.gripper_type not in [0, 1, 2, 3, 10] else args.gripper_type

    if args.gripper_speed is None:
        args.gripper_speed = 5000 if args.gripper_type == 1 else 225 if args.gripper_type == 2 else 4500 if args.gripper_type == 3 else 0
    else:
        args.gripper_speed = max(500, min(5000, args.gripper_speed)) if args.gripper_type == 1 else max(15, min(225, args.gripper_speed)) if args.gripper_type == 2 else max(500, min(4500, args.gripper_speed)) if args.gripper_type == 3 else 0
    if args.gripper_force is None:
        args.gripper_force = 0 if args.gripper_type == 1 else 50 if args.gripper_type == 2 else 100 if args.gripper_type == 3 else 0
    else:
        args.gripper_force = max(1, min(100, args.gripper_force))

    pika_to_robot_eef = [0, 0, 0, math.pi, -math.pi / 2, 0]

    logger.info('**********************************************************************')
    logger.info('* robot_ip: {}'.format(args.robot_ip))
    logger.info('* robot_mode: {}'.format(args.robot_mode))
    logger.info('* robot_speed: {}'.format(args.robot_speed))
    logger.info('* robot_acc: {}'.format(args.robot_acc))
    logger.info('* gripper_type: {}'.format(args.gripper_type))
    logger.info('* gripper_speed: {}'.format(args.gripper_speed))
    logger.info('* gripper_force: {}'.format(args.gripper_force))
    logger.info('* use_modbus: {}'.format(args.use_modbus))
    logger.info('* pika_to_robot: {}'.format(pika_to_robot_eef))
    logger.info('**********************************************************************')

    teleop = UFRobotTeleop(args, pika_to_robot_eef)
    teleop.run()

    # if len(sys.argv) < 2:
    #     print('Usage: {} {{robot_ip}} {{robot_mode}} {{gripper_type}}'.format(sys.argv[0]))
    #     print('  robot_mode: 1/7')
    #     print('     1: servo motion mode')
    #     print('     7: (default) cartesian online trajectory planning mode')
    #     print('  gripper_type: 0/1/2/3')
    #     print('     0: (default) no gripper')
    #     print('     1: xArm Gripper')
    #     print('     2: xArm Gripper G2')
    #     print('     3: BIO Gripper G2')
    #     exit(1)
    # robot_ip = sys.argv[1]
    # robot_mode = 7 if len(sys.argv) <= 2 else int(sys.argv[2])
    # robot_mode = robot_mode if robot_mode in [1, 7] else 7
    # # robot_mode==1: Servo模式
    # # robot_mode==7: 笛卡尔在线轨迹规划模式
    # gripper_type = 0 if len(sys.argv) <= 3 else int(sys.argv[3])
    # gripper_type = gripper_type if gripper_type in [0, 1, 2, 3] else 0
    # # gripper_type==0: 没有机械爪
    # # gripper_type==1: xArm Gripper
    # # gripper_type==2: xArm Gripper G2
    # # gripper_type==3: BIO Gripper G2
    # pika_to_robot_eef = [0, 0, 0, math.pi, -math.pi / 2, 0]
    # print('**********************************************************************')
    # print('* robot_ip: {}'.format(robot_ip))
    # print('* robot_mode: {}'.format(robot_mode))
    # print('* gripper_type: {}'.format(gripper_type))
    # print('* pika_to_robot: {}'.format(pika_to_robot_eef))
    # print('**********************************************************************')
    # teleop = UFRobotTeleop(robot_ip, pika_to_robot_eef, robot_mode=robot_mode, gripper_type=gripper_type)
    # teleop.run()
