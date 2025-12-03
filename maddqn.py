
from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
import Environment17
import os
from replay_memory import ReplayMemory  # must be the PER-enabled class
import matplotlib.pyplot as plt
import sys
import math

# 打印 GPU 状态信息
print("CUDA is available: ", tf.test.is_built_with_cuda())
print("CUDA runtime is available: ", tf.test.is_gpu_available())

# 设置日志级别及设备打印信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # INFO 级别
tf.debugging.set_log_device_placement(True)

# 配置 GPU 内存动态增长
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# 测试一个简单计算以触发设备分配
with tf.compat.v1.Session(config=config) as test_sess:
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = tf.add(a, b)
    print("Test computation result:", test_sess.run(c))

# 禁用 Eager Execution（采用 TF1 风格）
tf.compat.v1.disable_eager_execution()

# 重新设置 config
config = tf.compat.v1.ConfigProto()
tf.debugging.set_log_device_placement(True)
config.gpu_options.allow_growth = True

# 全局变量
reward = []

# ====== PER 超参数 ======
per_alpha = 0.6         
per_beta_start = 0.4
per_beta_end = 1.0
per_beta_current = per_beta_start
per_eps = 1e-6

# -------------------- 定义 Agent 类 --------------------
class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 0.99  # 折扣因子 (gamma)
        self.memory_entry_size = memory_entry_size
        # 使用支持 PER 的 ReplayMemory
        self.memory = ReplayMemory(self.memory_entry_size, memory_size=50000, batch_size=64, alpha=per_alpha, beta=per_beta_start)

# -------------------- 环境参数设置 --------------------
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
width = 750 / 2
height = 1298 / 2

IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN
label = 'marl_model'
n_veh = 4  # 参与决策车辆数量
n_neighbor = 4  # 每辆车考虑的邻居数量
n_RB = n_veh  # 资源块数量

# 初始化环境
env = Environment17.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game(n_veh)

# -------------------- 训练参数 --------------------
n_episode = 2000
n_step_per_episode = int(env.time_slow / env.time_fast)  # 每个 episode 的步数
epsi_final = 0.01  # 最终探索率
epsi_anneal_length = int(0.9 * n_episode)
mini_batch_step = n_step_per_episode  # mini-batch 训练间隔
target_update_step = n_step_per_episode * 4  # 目标网络更新步数

# -------------------- 状态获取函数 --------------------
def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    # 计算 V2I 快衰
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35
    # 计算 V2V 快衰
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] -
                env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]], np.newaxis] + 10) / 35
    # 计算 V2V 干扰
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # 计算绝对信道
    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0
    # 计算剩余负载和剩余时间（均归一化）
    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference,
                           np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([idx[0]])))

# -------------------- (新增) MHA 模块 --------------------
def _layer_norm(x, scope):
    with tf.compat.v1.variable_scope(scope):
        dim = x.get_shape()[-1]
        gamma = tf.compat.v1.get_variable('gamma', [dim], initializer=tf.ones_initializer())
        beta = tf.compat.v1.get_variable('beta', [dim], initializer=tf.zeros_initializer())
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return gamma * (x - mean) / tf.sqrt(var + 1e-6) + beta

def _build_tokens_from_flat_state(x_tensor, n_rb, n_veh):
    start_v2v_fast = n_rb
    end_v2v_fast = start_v2v_fast + n_veh * n_rb
    start_interf = end_v2v_fast
    end_interf = start_interf + n_rb
    v2i_fast = x_tensor[:, 0:n_rb]                              # [B, n_rb]
    v2v_interf = x_tensor[:, start_interf:end_interf]           # [B, n_rb]
    v2i_abs = x_tensor[:, end_interf:end_interf+1]              # [B, 1]
    scalars3 = x_tensor[:, end_interf+1+n_veh : end_interf+1+n_veh+3]  # [B,3]
    B = tf.shape(x_tensor)[0]
    scalars_all = tf.concat([v2i_abs, scalars3], axis=1)        # [B,4]
    scalars_tiled = tf.tile(tf.reshape(scalars_all, [B,1,4]), [1, n_rb, 1])  # [B,n_rb,4]
    rb_feat = tf.stack([v2i_fast, v2v_interf], axis=2)          # [B,n_rb,2]
    tokens = tf.concat([rb_feat, scalars_tiled], axis=2)        # [B,n_rb,6]
    return tokens

def _mha_block(tokens, num_heads, d_k, scope='mha'):
    with tf.compat.v1.variable_scope(scope):
        B = tf.shape(tokens)[0]
        T = int(tokens.get_shape()[1])
        Din = int(tokens.get_shape()[2])
        d_model = num_heads * d_k

        W_in = tf.compat.v1.get_variable('W_in', [Din, d_model], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b_in = tf.compat.v1.get_variable('b_in', [d_model], initializer=tf.zeros_initializer())
        H = tf.tensordot(tokens, W_in, axes=1) + b_in  # [B,T,d_model]

        Wq = tf.compat.v1.get_variable('Wq', [d_model, d_model], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        Wk = tf.compat.v1.get_variable('Wk', [d_model, d_model], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        Wv = tf.compat.v1.get_variable('Wv', [d_model, d_model], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        Wo = tf.compat.v1.get_variable('Wo', [d_model, d_model], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))

        def split_heads(X):
            X = tf.reshape(X, [B, T, num_heads, d_k])
            return tf.transpose(X, [0,2,1,3])  # [B,h,T,d_k]

        Q = tf.tensordot(H, Wq, axes=1)
        K_ = tf.tensordot(H, Wk, axes=1)
        V = tf.tensordot(H, Wv, axes=1)
        Qh, Kh, Vh = split_heads(Q), split_heads(K_), split_heads(V)
        scores = tf.matmul(Qh, Kh, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, Vh)  # [B,h,T,d_k]
        context = tf.transpose(context, [0,2,1,3])  # [B,T,h,d_k]
        context = tf.reshape(context, [B, T, num_heads*d_k])
        out = tf.tensordot(context, Wo, axes=1)  # [B,T,d_model]

        out = _layer_norm(out + H, scope='ln1')

        # FFN
        Wf1 = tf.compat.v1.get_variable('Wf1', [d_model, d_model*2], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        bf1 = tf.compat.v1.get_variable('bf1', [d_model*2], initializer=tf.zeros_initializer())
        Wf2 = tf.compat.v1.get_variable('Wf2', [d_model*2, d_model], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        bf2 = tf.compat.v1.get_variable('bf2', [d_model], initializer=tf.zeros_initializer())
        ffn = tf.nn.relu(tf.tensordot(out, Wf1, axes=1) + bf1)
        ffn = tf.tensordot(ffn, Wf2, axes=1) + bf2
        out = _layer_norm(out + ffn, scope='ln2')
        return out  # [B,T,d_model]

def _mha_preprocess(x_tensor, n_rb, n_veh, scope='mha_pre', proj_out_dim=None):
    with tf.compat.v1.variable_scope(scope):
        tokens = _build_tokens_from_flat_state(x_tensor, n_rb, n_veh)   # [B,n_rb,Dtok]
        attn = _mha_block(tokens, scope='block')                         # [B,n_rb,d_model]
        pooled = tf.reduce_mean(attn, axis=1)                            # [B,d_model]
        if proj_out_dim is None:
            proj_out_dim = int(x_tensor.get_shape()[1])
        Wp = tf.compat.v1.get_variable('Wp', [int(attn.get_shape()[-1]), proj_out_dim], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        bp = tf.compat.v1.get_variable('bp', [proj_out_dim], initializer=tf.zeros_initializer())
        proj = tf.matmul(pooled, Wp) + bp
        return x_tensor + proj

# -------------------- 构建统一的计算图 --------------------

n_input = len(get_state(env))
n_output = n_RB * len(env.V2V_power_dB_List)  # 动作数 = 资源块数 * 功率级别数

g = tf.Graph()
with g.as_default():
    # ---- 在线（训练）网络 ----
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="state")
    w_1 = tf.Variable(tf.compat.v1.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3, n_output], stddev=0.1))
    b_1 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.compat.v1.truncated_normal([n_output], stddev=0.1))

    # (新增) 在第一层前加一个 MHA 残差前端
    x_mha = _mha_preprocess(x, n_RB, n_veh, scope='mha_online', proj_out_dim=n_input)

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x_mha, w_1), b_1))
    layer_1_b = BatchNormalization()(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = BatchNormalization()(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = BatchNormalization()(layer_3)
    y = tf.add(tf.matmul(layer_3_b, w_4), b_4)  # 在线网络 Q(s,·)
    g_q_action = tf.argmax(y, axis=1, name="predicted_action")  # 在线网络选择 a* = argmax_a Q(s',a)

    # ---- 目标网络 ----
    x_target = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="target_state")
    w_1_t = tf.Variable(tf.compat.v1.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3, n_output], stddev=0.1))
    b_1_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_t = tf.Variable(tf.compat.v1.truncated_normal([n_output], stddev=0.1))

    # 目标网络同样加 MHA 前端（独立作用域）
    x_mha_t = _mha_preprocess(x_target, n_RB, n_veh, scope='mha_target', proj_out_dim=n_input)

    layer_1_t = tf.nn.relu(tf.add(tf.matmul(x_mha_t, w_1_t), b_1_t))
    layer_1_t_b = BatchNormalization()(layer_1_t)
    layer_2_t = tf.nn.relu(tf.add(tf.matmul(layer_1_t_b, w_2_t), b_2_t))
    layer_2_t_b = BatchNormalization()(layer_2_t)
    layer_3_t = tf.nn.relu(tf.add(tf.matmul(layer_2_t_b, w_3_t), b_3_t))
    layer_3_t_b = BatchNormalization()(layer_3_t)
    y_target = tf.add(tf.matmul(layer_3_t_b, w_4_t), b_4_t)  # 目标网络 Q'(s',·)

    # 为 target 网络选取特定索引的 Q 值（DDQN 用）
    g_target_q_idx = tf.compat.v1.placeholder(tf.int32, [None, None], name="target_q_idx")
    target_q_with_idx = tf.gather_nd(y_target, g_target_q_idx)

    # ====== (新增) PER 的 IS 权重占位符，并以加权 MSE 作为损失 ======
    g_target_q_t = tf.compat.v1.placeholder(tf.float32, [None], name="target_value")
    g_action = tf.compat.v1.placeholder(tf.int32, [None], name="action")
    g_is_w = tf.compat.v1.placeholder(tf.float32, [None], name='is_weight')

    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name="action_one_hot")
    q_acted = tf.reduce_sum(y * action_one_hot, axis=1, name="q_acted")
    td_err_vec = g_target_q_t - q_acted
    g_loss = tf.reduce_mean(g_is_w * tf.square(td_err_vec), name="g_loss")
    optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

# -------------------- DDQN + PER 批量训练函数（修改） --------------------
def q_learning_mini_batch_all(agents, current_sess, beta_for_per):

    # ---- 聚合所有 agent 的 batch ----
    batch_s_t_list, batch_s_t_plus_1_list = [], []
    batch_action_list, batch_reward_list = [], []
    batch_isw_list, batch_owner_slices, batch_idxs_all = [], [], []

    cursor = 0
    for ag in agents:
        # 用最新 beta 采样（带 is_weights 和 idxs）
        bs_t, bs_t1, ba, br, idxs, isw = ag.memory.sample(beta=beta_for_per)
        n_local = len(ba)
        batch_s_t_list.append(bs_t)
        batch_s_t_plus_1_list.append(bs_t1)
        batch_action_list.append(ba)
        batch_reward_list.append(br)
        batch_isw_list.append(isw)
        batch_idxs_all.append((ag, idxs, slice(cursor, cursor + n_local)))
        cursor += n_local

    batch_s_t = np.concatenate(batch_s_t_list, axis=0)
    batch_s_t_plus_1 = np.concatenate(batch_s_t_plus_1_list, axis=0)
    batch_action = np.concatenate(batch_action_list, axis=0)
    batch_reward = np.concatenate(batch_reward_list, axis=0)
    batch_isw = np.concatenate(batch_isw_list, axis=0)

    # ---- DDQN 目标：在线选 a*，目标网估值 ----
    pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
    q_t_plus_1 = current_sess.run(target_q_with_idx, feed_dict={
        x_target: batch_s_t_plus_1,
        g_target_q_idx: [[idx, a] for idx, a in enumerate(pred_action)]
    })
    gamma = agents[0].discount
    batch_target_q_t = batch_reward + gamma * q_t_plus_1

    # ---- 一次前向，得到 q_acted，用于 td_err 回写 ----
    q_acted_val = current_sess.run(q_acted, feed_dict={x: batch_s_t, g_action: batch_action})
    td_errors = np.abs(batch_target_q_t - q_acted_val) + per_eps

    # ---- 优化（加权 MSE） ----
    _, loss_val = current_sess.run([optim, g_loss], feed_dict={
        g_target_q_t: batch_target_q_t,
        g_action: batch_action,
        x: batch_s_t,
        g_is_w: batch_isw
    })

    # ---- 回写各自经验的优先级 ----
    for ag, idxs, sl in batch_idxs_all:
        ag_td = td_errors[sl]
        ag.memory.update_priorities(idxs, ag_td)

    return loss_val

# -------------------- 预测函数（ε-贪心） --------------------
def predict(current_sess, s_t, ep, test_ep=False):
    n_power_levels = len(env.V2V_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB * n_power_levels)
    else:
        pred_action = current_sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action

# -------------------- 目标网络参数硬更新 --------------------
def update_target_q_network(current_sess):
    update_ops = [
        w_1_t.assign(current_sess.run(w_1)),
        w_2_t.assign(current_sess.run(w_2)),
        w_3_t.assign(current_sess.run(w_3)),
        w_4_t.assign(current_sess.run(w_4)),
        b_1_t.assign(current_sess.run(b_1)),
        b_2_t.assign(current_sess.run(b_2)),
        b_3_t.assign(current_sess.run(b_3)),
        b_4_t.assign(current_sess.run(b_4)),
    ]
    current_sess.run(update_ops)

# -------------------- 初始化所有 Agent --------------------
agents = []
for ind_agent in range(n_veh * n_neighbor):
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

# 使用统一的 session 运行整个计算图
sess = tf.compat.v1.Session(graph=g, config=config)
sess.run(init)
print("Available GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

# -------------------- 训练过程 --------------------
record_reward = np.zeros([n_episode * n_step_per_episode, 1])
record_loss = []
Print_sum_reward = []
Print_Avg_V2I_SINR_C = []
Print_Avg_V2V_SINR_C = []
V2I_cdf = []
V2V_cdf = []

# 新增：用于记录每个 episode 的剩余负载和剩余时间预算（归一化）
Print_Avg_load_remaining = []
Print_Avg_time_remaining = []

num_repeats = 10  # 每个 episode 内重复训练次数
episode_loss_list = []

if __name__ == '__main__':
    for i_episode in range(n_episode):
        print("-------------------------")
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)
        else:
            epsi = epsi_final

        # β 退火（按 episode 线性退火至 1.0）
        frac = min(1.0, i_episode / float(max(1, n_episode - 1)))
        per_beta_current = per_beta_start + (per_beta_end - per_beta_start) * frac
        # 同步到每个 agent 的 memory（便于外部调用 sample(beta=None) 也能用最新 beta）
        for ag in agents:
            ag.memory.beta = per_beta_current

        # 每 100 个 episode 更新环境状态及信道信息
        if i_episode % 100 == 0:
            env.renew_vehicles_positions()
            env.renew_Vehicle_neighbor()


        repeat_rewards = []
        repeat_V2I_SINR = []
        repeat_V2V_SINR = []
        episode_temp_loss = []  # 保存当前 episode 内所有 repeat 的损失

        for repeat in range(num_repeats):
            # 重置环境需求、时间限制和激活链路
            env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
            env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
            env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

            sum_reward = 0
            epsi_V2I_SINR_C = 0
            epsi_V2V_SINR_C = 0
            repeat_loss = []  # 保存当前 repeat 内的所有 loss

            for i_step in range(n_step_per_episode):
                # 修改时间步计算，考虑 repeat 的影响
                time_step = i_episode * num_repeats * n_step_per_episode + repeat * n_step_per_episode + i_step
                state_old_all1 = []
                action_all1 = []
                action_all_training1 = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

                # 对每个车辆及其邻居选择动作
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        state = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                        state_old_all1.append(state)
                        action = predict(sess, state, epsi)
                        action_all1.append(action)
                        action_all_training1[i, j, 0] = i  # 记录车辆索引（作为资源块编号）
                        action_all_training1[i, j, 1] = int(np.floor(action / n_RB))  # 功率级别

                action_temp = action_all_training1.copy()
                # 执行动作，获得奖励和 SINR 指标
                V2I_SINR_C, V2V_SINR_C, train_reward = env.act_for_training(action_temp)
                if i_episode > 1000:
                    V2V_SINR_C = 150 + (150 - V2V_SINR_C)
                sum_reward += train_reward
                epsi_V2I_SINR_C = V2I_SINR_C
                epsi_V2V_SINR_C = V2V_SINR_C
                # 更新快衰信道


                # 存储每条链路的经验（初始优先级默认使用当前最大优先级）
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        state_old = state_old_all1[n_neighbor * i + j]
                        actions = action_all1[n_neighbor * i + j]
                        state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                        agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, actions)

                # 每隔一定步数进行一次批量训练和目标网络更新
                if i_episode > 500:
                    if time_step % mini_batch_step == mini_batch_step - 1:
                        loss_val_batch = q_learning_mini_batch_all(agents, sess, per_beta_current)
                        repeat_loss.append(loss_val_batch)
                    if time_step % target_update_step == target_update_step - 1:
                        update_target_q_network(sess)
            # 计算当前 repeat 的平均 loss（仅当有训练时）
            if len(repeat_loss) > 0:
                avg_repeat_loss = np.mean(repeat_loss)
                episode_temp_loss.append(avg_repeat_loss)

            repeat_rewards.append(sum_reward)
            repeat_V2I_SINR.append(epsi_V2I_SINR_C)
            repeat_V2V_SINR.append(epsi_V2V_SINR_C)
            print(f"Episode: {i_episode}, Repeat: {repeat}, Reward: {round(sum_reward, 4)}, "
                  f"Avg V2I SINR C: {round(epsi_V2I_SINR_C, 4)}, Avg V2V SINR C: {round(epsi_V2V_SINR_C, 4)}")

        # 计算当前 episode 的平均 loss（仅当有 valid loss 时）
        if len(episode_temp_loss) > 0:
            avg_episode_loss = np.mean(episode_temp_loss)
        else:
            avg_episode_loss = 0  # 未训练时设为 0
        episode_loss_list.append(avg_episode_loss)


        print(f"Episode: {i_episode}, Explore: {round(epsi, 4)}, Avg Reward: {round(Print_sum_reward[-1], 4)} | PER beta={per_beta_current:.3f}")
        print(f"Avg V2I SINR: {round(Print_Avg_V2I_SINR_C[-1], 4)}, Avg V2V SINR: {round(Print_Avg_V2V_SINR_C[-1], 4)}")
        print(f"Avg Load Remaining: {round(avg_load_remaining, 4)}, Avg Time Remaining: {round(avg_time_remaining, 4)}")

    # 绘制结果图（累计奖励、V2I/V2V SINR、剩余负载、剩余时间）
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 5, 1)
    plt.plot(range(n_episode), Print_sum_reward, label="Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Cumulative Reward per Episode")
    plt.legend()

    plt.subplot(1, 5, 2)
    plt.plot(range(n_episode), Print_Avg_V2I_SINR_C, label="Avg V2I SINR", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("SINR")
    plt.title("Average V2I SINR per Episode")
    plt.legend()

    plt.subplot(1, 5, 3)
    plt.plot(range(n_episode), Print_Avg_V2V_SINR_C, label="Avg V2V SINR", color='green')
    plt.xlabel("Episode")
    plt.ylabel("SINR")
    plt.title("Average V2V SINR per Episode")
    plt.legend()


    plt.tight_layout()
    plt.show()

    # 绘制每个 episode 的平均损失曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(episode_loss_list, label="Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Loss Function per Episode")
    plt.legend()
    plt.tight_layout()
    plt.show()


    sess.close()

