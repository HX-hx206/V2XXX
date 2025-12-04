# 修改奖励函数
from __future__ import division
import numpy as np
import time
import random
import math
# from scipy.special import jv  # 贝塞尔函数
from scipy.special import j0
import scipy
import os

class HumanDrivenVehicle:
    def __init__(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
        self.agent_id = agent_id
        self.max_power = max_power
        self.current_power = np.random.rand() * max_power
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
        self.type = type_id

    def take_action(self):
        # 随机选择一个发射功率,后根据策略更新
        self.current_power = np.random.rand() * self.max_power
        return self.current_power

class AutonomousDrivenVehicle:
    def __init__(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
        self.agent_id = agent_id
        self.max_power = max_power
        self.current_power = np.random.rand() * max_power
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
        self.type = type_id

    def take_action(self):
        # 随机选择一个发射功率，后根据策略更新
        self.current_power = np.random.rand() * self.max_power
        return self.current_power

class V2Vchannels:
    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3
        self.G = 70
        self.alpha = 3
        self.c = 3e8
        self.update_time_interval = 0.1


    def get_distance(self, position_A, position_B):
        """计算两点间的距离"""
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        return math.hypot(d1, d2) + 0.001

    '''计算大尺度衰落'''
    def get_shadowing(self, delta_distance, shadowing):
        """计算对数正态分布的阴影衰落增益"""
        # shadowings = np.array(shadowing)
        shadowing_dB = (
                np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing +
                math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) *
                np.random.normal(0, self.shadow_std)
        )
        shadowing_linear = 10 ** (shadowing_dB / 10)

        return shadowing_linear

    def get_large_scale_fading(self, position_A, position_B, shadowing):
        """计算大尺度衰落 L_{x,y}(t)"""
        d = self.get_distance(position_A, position_B)
        shadowing_linear = self.get_shadowing(d, shadowing)
        large_scale_fading = self.G * shadowing_linear / (d ** self.alpha)
        return large_scale_fading

class V2Ichannels:
    def __init__(self):
        self.t = 0
        self.h_bs = 25
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]
        self.shadow_std = 8
        self.G = 70
        self.alpha = 3
        self.c = 3e8
        self.v = 50
        self.update_time_interval = 0.1



    def get_distance(self, position_A):
        """计算两点间的距离"""
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        return math.hypot(d1, d2) + 0.001

    '''计算大尺度衰落'''
    def get_shadowing(self, delta_distance, shadowing):
        """计算对数正态分布的阴影衰落增益"""

        shadowing_dB = (
                np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing +
                math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) *
                np.random.normal(0, self.shadow_std)
        )
        shadowing_linear = 10 ** (shadowing_dB / 10)
        return shadowing_linear

    def get_large_scale_fading(self, position_A, shadowing):
        """计算大尺度衰落 L_{x,y}(t)"""
        d = self.get_distance(position_A)
        shadowing_linear = self.get_shadowing(d, shadowing)
        large_scale_fading = self.G * shadowing_linear / (d ** self.alpha)
        return large_scale_fading

class Environ:

        def __init__(self, down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor):

            self.down_lanes = down_lanes
            self.up_lanes = up_lanes
            self.left_lanes = left_lanes
            self.right_lanes = right_lanes
            self.width = width
            self.height = height

            '''信道和车辆定义'''
            self.V2Vchannels = V2Vchannels()
            self.V2Ichannels = V2Ichannels()
            self.HumanDrivenVehicle_list = []
            self.AutonomousDrivenVehicle_list = []
            self.vehicles = []

            '''状态和信道参数'''
            self.demand = []
            self.V2V_Shadowing = []
            self.V2I_Shadowing = []
            self.delta_distance = []
            self.V2V_channels_abs = []
            self.V2I_channels_abs = []

            '''通信功率和噪声'''
            self.update_time_interval = 0.1
            self.V2I_power_dB = 46
            self.V2V_power_dB_List = [23, 20, 8, 5,-100]
            self.sig2_dB = -114
            self.bsAntGain = 10
            self.bsNoiseFigure = 5
            self.vehAntGain = 3
            self.vehNoiseFigure = 9
            self.sig2 = 10 ** (self.sig2_dB / 10)
            self.f_c = 2e9
            self.c = 3e8

            '''资源块和车辆设置'''
            self.n_RB = n_veh * 2
            # self.n_RB = 1
            self.n_Veh = n_veh * 2
            self.n_neighbor = n_neighbor

            '''时间和带宽设置'''
            self.time_fast = 0.001
            self.time_slow = 0.1
            self.bandwidth = int(1e6)

            '''需求和干扰设置'''
            self.demand_size = int((4 * 190 + 300) * 8 * 1)
            self.V2V_Interference_all = np.zeros((self.n_Veh * 2, self.n_neighbor, self.n_RB)) + self.sig2

    ######################为环境添加车辆（初始化场景中的车辆）###############################################


        '''添加有人驾驶汽车'''
        def add_new_HumanDrivenVehicle(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
            self.vehicles.append(HumanDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))
            self.HumanDrivenVehicle_list.append(HumanDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))

        '''添加无人驾驶汽车'''
        def add_new_AutonomousDrivenVehicle(self, agent_id, max_power, start_position, start_direction, velocity, type_id):
            self.vehicles.append(AutonomousDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))
            self.AutonomousDrivenVehicle_list.append(AutonomousDrivenVehicle(agent_id, max_power, start_position, start_direction, velocity, type_id))

        '''随机添加有人驾驶汽车'''
        def add_new_HumanDrivenVehicle_by_number(self, n):

            for i in range(n):
                agent_id = i
                max_power = 23
                # 随机选择下行车道和起始位置
                ind = np.random.randint(0, len(self.down_lanes))
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'd' # 设定初始方向为向下
                # self.add_new_HumanDrivenVehicle(agent_id, max_power, start_position, start_direction, np.random.randint(10, 15),"Human")# 每辆新车辆的速度在10到15米/秒之间随机。
                self.add_new_HumanDrivenVehicle(agent_id, max_power, start_position, start_direction,
                                                80, "Human")  # 每辆新车辆的速度在10到15米/秒之间随机。

                # 在上行车道添加车辆
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'u'
                self.add_new_HumanDrivenVehicle(agent_id + 1, max_power,start_position, start_direction, 80,"Human")

                # 在左行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
                start_direction = 'l'
                self.add_new_HumanDrivenVehicle(agent_id + 2, max_power,start_position, start_direction, 80,"Human")

                # 在右行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
                start_direction = 'r'
                self.add_new_HumanDrivenVehicle(agent_id + 3, max_power,start_position, start_direction, 80,"Human")

            self.new_channelmodel()
            # # 初始化车辆间通信的信道模型参数
            # self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
            # self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
            #
            # # 计算车辆间的距离变化量
            # self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

        '''随机添加无人驾驶汽车'''
        def add_new_AutonomousDrivenVehicle_by_number(self, n):

            for i in range(n):
                agent_id = 1
                max_power = 100
                # 随机选择下行车道和起始位置
                ind = np.random.randint(0, len(self.down_lanes))
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'd' # 设定初始方向为向下
                self.add_new_AutonomousDrivenVehicle(agent_id, max_power,start_position, start_direction, 80,"Auto")# 每辆新车辆的速度在10到15米/秒之间随机。

                # 在上行车道添加车辆
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
                start_direction = 'u'
                self.add_new_AutonomousDrivenVehicle(agent_id + 1, max_power,start_position, start_direction, 80,"Auto")

                # 在左行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
                start_direction = 'l'
                self.add_new_AutonomousDrivenVehicle(agent_id + 2, max_power,start_position, start_direction, 80,"Auto")

                # 在右行车道添加车辆
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
                start_direction = 'r'
                self.add_new_AutonomousDrivenVehicle(agent_id + 3, max_power,start_position, start_direction, 80,"Auto")

            self.new_channelmodel()
        '''初始化车辆之间通信的信道模型参数'''
        def new_channelmodel(self):
            # 初始化车辆间通信的信道模型参数
            # self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
            # self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
            self.V2V_Shadowing = np.random.normal(0, 3)
            self.V2I_Shadowing = np.random.normal(0, 8)

            # 计算车辆间的距离变化量
            self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])

        '''计算小尺度衰落信道增益（多普勒）'''
        def apply_doppler_effect(self, h, velocity_A, velocity_B):
            """
            参数:
            - h: 上一时刻的小尺度衰落信道增益 (复数，numpy array)
            - velocity_A: 车辆 A 的速度向量 [v_x, v_y]
            - velocity_B: 车辆 B 的速度向量 [v_x, v_y]
            return：
            - h_next：此时刻的小尺度衰落信道增益
            """

            # 计算相对速度的大小
            relative_velocity = np.array(velocity_B) - np.array(velocity_A)
            relative_speed = np.linalg.norm(relative_velocity)  # 相对速度大小

            # 计算最大多普勒频移
            max_doppler_shift = (relative_speed / self.c) * self.f_c

            # 计算零阶贝塞尔函数 J_0(2πf_D T)
            epsilon = j0(2 * np.pi * max_doppler_shift * self.update_time_interval)

            # 小尺度衰落的随机分量 (复数高斯分布)
            eta_real = np.random.normal(0, 1)
            eta_imag = np.random.normal(0, 1)
            e = (eta_real + 1j * eta_imag) / np.sqrt(2)  # CN(0, 1)

            # 计算下一时刻信道增益
            h_next = epsilon * h + np.sqrt(1 - epsilon ** 2) * e
            return np.abs(h_next)


    ######################       更新车辆位置          ###########################################
        '''更新车辆位置：遍历每辆车，根据其方向和速度更新位置'''
        def renew_vehicles_positions(self):
            i = 0
            while (i < len(self.vehicles)):
                delta_distance = self.vehicles[i].velocity * self.time_slow         #计算车辆在此时间步长内行驶的距离。车辆速度与时间步长的乘积
                change_direction = False                                                      #车辆在当前更新周期内是否改变方向

                '''如果车辆向上移动（u），检查它是否应该改变方向。'''
                if self.vehicles[i].direction == 'u':
                    #检查左车道十字路口（上行）
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至left（l）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    #检查右车道十字路口（上行）
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至right（r）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    #如果没有改变则增加垂直位置
                    if change_direction == False:
                        self.vehicles[i].position[1] += delta_distance

                '''如果车辆向下移动（d），检查它是否应该改变方向。'''
                if (self.vehicles[i].direction == 'd') and (change_direction == False):
                    #检查左车道十字路口（下行）
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    # 该模块检查车辆在（left_lane）向下移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至left（l）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    #检查右车道十字路口（下行）
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至right（r）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 如果没有改变则减去垂直位置
                    if change_direction == False:
                        self.vehicles[i].position[1] -= delta_distance

                '''如果车辆向右移动（r），检查它是否应该改变方向。'''
                if (self.vehicles[i].direction == 'r') and (change_direction == False):
                    # 检查左车道十字路口（右行）
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    # 该模块检查车辆在（left_lane）向下移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至up（u）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 检查右车道十字路口（右行）
                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                    #该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至down（d）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 如果没有改变则继续向右移动，加上去
                    if change_direction == False:
                        self.vehicles[i].position[0] += delta_distance

                '''如果车辆向左移动（l），检查它是否应该改变方向。'''
                if (self.vehicles[i].direction == 'l') and (change_direction == False):
                    # 检查左车道十字路口（左行）
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    # 该模块检查车辆在（left_lane）向下移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至up（u）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < 0.4):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                    # 该模块检查车辆在（left_lane）向上移动过程中是否会穿过（十字路口），如果车辆穿过车道则有40 % 的可能性它会改变方向至down（d）如果它改变方向，则更新其位置并将change_direction标志设置为True。

                    # 如果没有改变则继续向左移动，减去
                    if change_direction == False:
                            self.vehicles[i].position[0] -= delta_distance

                '''处理退出'''
                #如果超出模拟边界则重新定位
                if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):

                    #改变方向
                    if (self.vehicles[i].direction == 'u'):
                        self.vehicles[i].direction = 'r'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                    else:
                        if (self.vehicles[i].direction == 'd'):
                            self.vehicles[i].direction = 'l'
                            self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                        else:
                            if (self.vehicles[i].direction == 'l'):
                                self.vehicles[i].direction = 'u'
                                self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                            else:
                                if (self.vehicles[i].direction == 'r'):
                                    self.vehicles[i].direction = 'd'
                                    self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

                '''移至列表中的下一辆车'''
                i += 1

    ######################       更新车辆邻居          ###########################################
        '''更新汽车的邻居'''
        def renew_Vehicle_neighbor(self):
            for i in range(len(self.vehicles)):
                self.vehicles[i].neighbors = []     #将第 i 辆车的 neighbors 属性重置为空列表，表示每辆车的邻居需要重新计算。
                self.vehicles[i].actions = []       #将第 i 辆车的 actions 属性重置为空列表，通常表示每辆车的可选动作集需要在之后进行更新。

            #对于每辆车，将其二维位置 (x, y) 转换为一个复数 x + yi，其中 x 对应复数的实部，y 对应虚部。这种表示形式可以简化距离计算。
            z1 = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])

            #abs计算复数差值的绝对值，即得到每辆车与其他车之间的欧几里得距离。Distance 是一个二维矩阵，其中 Distance[i][j] 表示第 i 辆车与第 j 辆车之间的距离。
            Distance = abs(z1.T - z1)

            #查找每辆车最近的邻居
            for i in range(len(self.vehicles)):
                sort_idx = np.argsort(Distance[:, i])

                #选择前 n_neighbor 个最近的邻居
                for j in range(self.n_neighbor):
                    self.vehicles[i].neighbors.append(sort_idx[j + 1])

                #将车辆的目的地设置为它的邻居列表
                destination = self.vehicles[i].neighbors
                self.vehicles[i].destinations = destination


    ######################       更新车辆信道          ###########################################
        '''更新车辆信道'''
        def renew_vehicles_channel(self,h, velocity_A, velocity_B):
            #初始化 V2V 和 V2I 信道损耗矩阵
            self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
            self.V2I_pathloss = np.zeros((len(self.vehicles)))

            #初始化 V2V 和 V2I 信道特性的绝对值矩阵
            self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
            self.V2I_channels_abs = np.zeros((len(self.vehicles)),)


            #计算车辆间（V2V）信道的阴影效应和路径损耗
            for i in range(len(self.vehicles)):
                for j in range(i + 1, len(self.vehicles)):
                    # 计算大尺度衰落
                    # self.V2V_pathloss[j][i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_large_scale_fading(self.vehicles[i].position, self.vehicles[j].position, self.V2V_Shadowing)
                    self.V2V_pathloss[j][i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_large_scale_fading(
                        self.vehicles[i].position, self.vehicles[j].position, 1)
                    pathloss_linear = 10 ** (-self.V2V_pathloss[i][j] / 10)  # 将 dB 转为线性值

                    #计算小尺度衰落
                    h = self.apply_doppler_effect(h, velocity_A, velocity_B)
                    self.V2V_channels_abs[i][j] = self.V2V_channels_abs[j][i] = np.abs(h) ** 2 * pathloss_linear

            # 计算并更新V2I路径损耗和阴影衰落
            for i in range(len(self.vehicles)):
                #计算大尺度衰落
                self.V2I_pathloss[i] = self.V2Ichannels.get_large_scale_fading(self.vehicles[i].position, self.V2Ichannels.shadow_std)
                pathloss_linear = 10 ** (-self.V2V_pathloss[i]/ 10)  # 将 dB 转为线性值

                # 生成小尺度衰落
                h = self.apply_doppler_effect(h, velocity_A, velocity_B)
                h_V2I = np.abs(h) ** 2 * pathloss_linear
                self.V2I_channels_abs[i]= h_V2I[i]




        '''更新 V2V 信道的快速衰落'''
        def renew_channels_fastfading(self, h, velocity_A, velocity_B):

                '''更新 V2V 信道的快速衰落'''
                V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)

                # 多普勒频移
                f_d = self.apply_doppler_effect(h, velocity_A, velocity_B)  # 最大多普勒频移 (Hz)

                # 计算时间相关系数
                correlation_coefficient = scipy.special.j0(2 * np.pi * f_d * self.update_time_interval)

                # 快速衰落的复高斯分布
                fading_real = np.random.normal(0, 1, V2V_channels_with_fastfading.shape)
                fading_imag = np.random.normal(0, 1, V2V_channels_with_fastfading.shape)

                # 叠加时间相关性
                if hasattr(self, 'V2V_channels_fastfading_prev'):  # 如果是非首次调用
                    self.V2V_channels_fastfading = (
                            correlation_coefficient * self.V2V_channels_fastfading_prev
                            + math.sqrt(1 - correlation_coefficient ** 2)
                            * (fading_real + 1j * fading_imag) / math.sqrt(2)
                    )
                else:  # 第一次调用，初始化信道
                    self.V2V_channels_fastfading = (fading_real + 1j * fading_imag) / math.sqrt(2)





        ######################       计算奖励          ###########################################
        '''计算性能奖励（********问题**********）'''
        '''训练阶段，计算 V2I 和 V2V 通信的性能（速率、干扰）和奖励'''

        def Compute_Performance_Reward_Train(self, actions_power):

            actions = actions_power[:, :, 0].astype(int)  # 每个 V2V 链路使用的 RB (0..n_RB-1)
            Psel = actions_power[:, :, 1]


            nRB = self.n_RB
            nVeh = len(self.vehicles)
            nNei = self.n_neighbor



            consider_v2i_to_v2v_interf = False


            self.SV2I_channels_with_fastfading = self.V2I_channels_with_fastfading.mean(axis=1)

            # ---------------- 容器 ----------------
            V2I_Signals = np.zeros(nRB)  # 每个 RB 的 V2I 信号功率（线性）
            V2I_Interference = np.zeros(nRB)  # 每个 RB 的 V2I 干扰（来自复用 V2V）
            V2V_Signals = np.zeros((nVeh, nNei))  # 每条 V2V 链路的信号
            V2V_Interference = np.zeros((nVeh, nNei))  # 每条 V2V 链路的干扰


            V2I_Signals = 10 ** ((self.V2I_power_dB - np.diag(self.V2I_channels_with_fastfading)
                                  + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10.0)

            # ---------------- 遍历每个 RB，把复用该 RB 的 V2V 链路找出来 ----------------


       
            # ---------------- 奖励（示例：线性加权） ----------------
            total_reward = α * human_mean_cap + β * auto_mean_cap + γ * V2I_reward


            return V2I_Capacity, V2V_Capacity, total_reward, human_mean_cap, auto_mean_cap, V2I_reward

        def act_for_training(self, actions):
            action_temp = actions.copy()


            V2I_C, V2V_C, total_rewards, HumanDriven_rewards, Autonomous_rewards, V2I_rewards = \
                self.Compute_Performance_Reward_Train(action_temp)


            rows = min(V2V_C.shape[0], self.active_links.shape[0])
            valid_mask = self.active_links[:rows, :]
            success_matrix = (V2V_C[:rows, :] >= getattr(self, 'R_V2V_min', 1e5)) & valid_mask

            denom = max(1, valid_mask.sum())
            V2V_success = success_matrix.sum() / denom





            V2I_SINR_C_disp = (np.mean(V2I_C)) / (10 ** 5)
            V2V_SINR_C_disp = (np.mean(V2V_C[:rows, :])) / (10 ** 6)


            return V2I_SINR_C_disp, V2V_SINR_C_disp, total_rewards

        ######################       初始化场景          ###########################################
        '''初始化一个新的随机场景'''
        def new_random_game(self, n_Veh=0):
            #初始化车辆列表
            self.HumanDrivenVehicle_list = []
            self.AutonomousDrivenVehicle_list = []
            self.vehicles = []

            








