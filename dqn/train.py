import os
import random
import pickle
import json
import multiprocessing as mp
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----------------- 1. 导入编译好的 C++ 扩展 -----------------
import mcts_cpp

# ----------------- 2. CONFIG -----------------
CONFIG = {
    "BOARD_ROWS": 5,
    "BOARD_COLS": 5,
    "MAX_STACK": 5,
    "RESIDUAL_BLOCKS": 10,
    "RES_FILTERS": 320,
    "POLICY_SIZE": 25,
    "NUM_WORKERS": 2,
    "MCTS_SIMS": 6000,
    "GAMES_PER_WORKER": 10,
    "EPISODES": 2500,
    "UPDATE_FREQ": 20,
    "BATCH_SIZE": 256,
    "REPLAY_BUFFER_SIZE": 50000,
    "LEARNING_RATE": 3e-5,
    "TEMP_INIT": 1.0,
    "TEMP_FINAL": 0.01,
    "TEMP_DECAY_EPISODES": 2000,
    "SAVE_INTERVAL": 50,
    "CHECKPOINT_DIR": "./ckpt/",  # 存放权重/日志等
    "REPLAY_FILE": "replay_buffer.pkl",
    "TRAIN_STATE_FILE": "train_state.json",
    "EVAL_GAMES": 1,
    "MINIMAX_DEPTH": 5,
    "LOG_FILE": "train.log",
}

os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)


# ----------------- 3. 网络定义 -----------------
class ResidualBlock(nn.Module):
    def __init__(self, filters=128):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop(x)
        x += shortcut
        x = self.relu(x)
        return x


class AlphaZeroResNet(nn.Module):
    def __init__(
        self, board_size=(5, 5), num_res_blocks=10, filters=320, policy_size=25
    ):
        super().__init__()
        self.conv_init = nn.Conv2d(2, filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(filters)
        self.relu_init = nn.ReLU(inplace=True)

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(filters))
        self.res_blocks = nn.Sequential(*blocks)

        # policy head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * board_size[0] * board_size[1], policy_size)

        # value head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(board_size[0] * board_size[1], 128)
        self.value_drop = nn.Dropout(0.1)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        x = self.res_blocks(x)

        # policy
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.reshape(p.size(0), -1)
        logits = self.policy_fc(p)

        # value
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_relu(v)
        v = v.reshape(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_drop(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)
        return logits, v


# ----------------- 4. Replay Buffer -----------------
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def push(self, data_list):
        """
        data_list: list of (obs, pi, z), each is one step data
        """
        for d in data_list:
            self.buffer.append(d)

    def sample(self, batch_size, augment=True):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        obs_list = [b[0] for b in batch]  # shape=(5,5,2)
        pi_list = [b[1] for b in batch]  # shape=25
        z_list = [b[2] for b in batch]

        obs_np = np.array(obs_list, dtype=np.float32)
        pi_np = np.array(pi_list, dtype=np.float32)
        z_np = np.array(z_list, dtype=np.float32)

        obs_t = torch.from_numpy(obs_np)  # (B,5,5,2)
        pi_t = torch.from_numpy(pi_np)  # (B,25)
        z_t = torch.from_numpy(z_np)  # (B,)

        return (obs_t, pi_t, z_t)

    def __len__(self):
        return len(self.buffer)


# ----------------- 6. 训练 batch -----------------
def train_on_batch(model, optimizer, states, pis, zs, device):
    model.train()
    states = states.permute(0, 3, 1, 2).to(device)  # (B,2,5,5)
    pis = pis.to(device)  # (B,25)
    zs = zs.to(device)  # (B,)

    optimizer.zero_grad()
    logits, values = model(states)
    policy = F.log_softmax(logits, dim=1)  # (B,25)
    policy_loss = -torch.mean(torch.sum(pis * policy, dim=1))
    values = values.squeeze(1)
    value_loss = F.mse_loss(values, zs)

    reg = 0
    for p in model.parameters():
        reg += torch.sum(p**2)
    loss = policy_loss + value_loss + 1e-4 * reg
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


# ----------------- 7. 子进程: 调用 c++ run_self_play_once_alphaZero -----------------
def self_play_worker(pid, param_queue, data_queue, config):
    local_model = AlphaZeroResNet(
        board_size=(config["BOARD_ROWS"], config["BOARD_COLS"]),
        num_res_blocks=config["RESIDUAL_BLOCKS"],
        filters=config["RES_FILTERS"],
        policy_size=config["POLICY_SIZE"],
    )
    local_model.eval()
    worker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(worker_device)

    def worker_infer(obs_3d):
        """
        C++在Expansion阶段回调此函数:
        obs_3d shape=(5,5,2)
        return (logits(25), value)
        """
        obs_np = np.array(obs_3d, dtype=np.float32)
        x = torch.from_numpy(obs_np).unsqueeze(0).permute(0, 3, 1, 2).to(worker_device)
        with torch.no_grad():
            lg, val = local_model(x)
        lg_list = lg[0].cpu().numpy().tolist()
        valf = float(val[0].cpu().item())
        return (lg_list, valf)

    while True:
        message = param_queue.get()
        if message is None:
            # 主进程发来的结束信号
            break

        # 解包 (weights, temperature)
        weights, temperature = message
        local_model.load_state_dict(weights)

        batch_data = []
        for _ in range(config["GAMES_PER_WORKER"]):
            data = mcts_cpp.run_self_play_once_alphaZero(
                worker_infer,  # Python函数: local NN forward
                config["MCTS_SIMS"],  # mcts_sims
                1.0,  # c_puct
                temperature,  # temp
            )
            # data is list of (obs, pi, z)
            batch_data.extend(data)

        data_queue.put(batch_data)


# ----------------- 9. MAIN TRAIN LOOP -----------------
def main():
    config = CONFIG
    os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)
    log_file = os.path.join(config["CHECKPOINT_DIR"], config["LOG_FILE"])
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Training Log\n\n")

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available")
    model = AlphaZeroResNet(
        board_size=(config["BOARD_ROWS"], config["BOARD_COLS"]),
        num_res_blocks=config["RESIDUAL_BLOCKS"],
        filters=config["RES_FILTERS"],
        policy_size=config["POLICY_SIZE"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    replay = ReplayBuffer(config["REPLAY_BUFFER_SIZE"])

    # spawn
    ctx = mp.get_context("spawn")
    param_queue = ctx.Queue()
    data_queue = ctx.Queue()
    workers = []
    for w in range(config["NUM_WORKERS"]):
        p = ctx.Process(
            target=self_play_worker, args=(w, param_queue, data_queue, config)
        )
        p.start()
        workers.append(p)

    start_episode = 0
    replay_file = os.path.join(config["CHECKPOINT_DIR"], config["REPLAY_FILE"])
    train_state_file = os.path.join(
        config["CHECKPOINT_DIR"], config["TRAIN_STATE_FILE"]
    )

    # 如果有历史,尝试加载
    if os.path.exists(replay_file) and os.path.exists(train_state_file):
        with open(replay_file, "rb") as f:
            try:
                data = pickle.load(f)
                if isinstance(data, list):
                    replay.buffer.extend(data)
                    print(f"[Resume] Replay loaded. size={len(replay)}")
            except Exception as e:
                print(f"[Resume] Fail replay load: {e}")
        with open(train_state_file, "r") as f:
            try:
                st = json.load(f)
                start_episode = st.get("episode", 0)
                last_ckpt = st.get("last_ckpt", None)
                if last_ckpt and os.path.exists(last_ckpt):
                    model.load_state_dict(torch.load(last_ckpt, map_location=device))
                    print(f"[Resume] Model weights loaded from {last_ckpt}")
                print(f"[Resume] Start from ep={start_episode+1}")
            except Exception as e:
                print(f"[Resume] Fail state load: {e}")
    else:
        print("[Info] Start from scratch.")

    # minimax_ai = MinimaxAI(depth=config["MINIMAX_DEPTH"])

    for ep in range(start_episode, config["EPISODES"]):
        # 从EP=750开始,温度线性衰减到0.01
        frac = min(1.0, (ep - 750) / (config["TEMP_DECAY_EPISODES"] - 750 + 1))
        temperature = config["TEMP_INIT"] * (1 - frac) + config["TEMP_FINAL"] * frac

        # 下发当前权重
        wts = model.state_dict()
        for _ in range(config["NUM_WORKERS"]):
            param_queue.put((wts, temperature))

        # 收集自对弈数据
        for _ in range(config["NUM_WORKERS"]):
            data = data_queue.get()
            replay.push(data)

        # 训练
        total_loss, total_pl, total_vl = 0.0, 0.0, 0.0
        for _ in range(config["UPDATE_FREQ"]):
            batch = replay.sample(config["BATCH_SIZE"])
            if batch is None:
                continue
            s, pi, z = batch
            ls, pl, vl = train_on_batch(model, optimizer, s, pi, z, device)
            total_loss += ls
            total_pl += pl
            total_vl += vl
        total_loss /= max(1, config["UPDATE_FREQ"])
        total_pl /= max(1, config["UPDATE_FREQ"])
        total_vl /= max(1, config["UPDATE_FREQ"])

        cur_ep = ep + 1
        if (cur_ep % config["SAVE_INTERVAL"]) == 0:
            ckpt_path = os.path.join(config["CHECKPOINT_DIR"], f"ckpt_ep{cur_ep}.pth")
            torch.save(model.state_dict(), ckpt_path)
            # dump replay
            with open(replay_file, "wb") as f:
                pickle.dump(list(replay.buffer), f)
            train_state = {"episode": cur_ep, "last_ckpt": ckpt_path}
            with open(train_state_file, "w") as f:
                json.dump(train_state, f)

            # 评估
            # start_time = datetime.now()
            # wr, aWins, mWins, dr = evaluate_model(
            #     model, minimax_ai, device, config["EVAL_GAMES"]
            # )
            # elapsed = (datetime.now() - start_time).total_seconds()
            # model_name = f"ckpt_ep{cur_ep}.pth"
            # logObj = {
            #     "model_name": model_name,
            #     "match_number": cur_ep // config["SAVE_INTERVAL"],
            #     "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            #     "elapsed_time_sec": elapsed,
            #     "win_rate": wr,
            #     "ai_wins": aWins,
            #     "minimax_wins": mWins,
            #     "draws": dr,
            # }
            # log_path = os.path.join(config["CHECKPOINT_DIR"], config["LOG_FILE"])
            # with open(log_path, "a") as f:
            #     f.write(json.dumps(logObj) + "\n")

            # print(
            #     f"[EP {cur_ep}] Saved ckpt: {ckpt_path}, Loss={total_loss:.4f}, P={total_pl:.4f}, V={total_vl:.4f}, WinRate={wr*100:.2f}%"
            # )
            print(
                f"[EP {cur_ep}] Buffer={len(replay)}, T={temperature:.2f}, Loss={total_loss:.4f}, Policy Loss={total_pl:.4f}, Value Loss={total_vl:.4f}"
            )
        else:
            print(
                f"[EP {cur_ep}] Buffer={len(replay)}, T={temperature:.2f}, Loss={total_loss:.4f}, Policy Loss={total_pl:.4f}, Value Loss={total_vl:.4f}"
            )

    # 收尾
    for _ in range(config["NUM_WORKERS"]):
        param_queue.put(None)
    for p in workers:
        p.join()

    final_ckpt = os.path.join(config["CHECKPOINT_DIR"], "final.pth")
    torch.save(model.state_dict(), final_ckpt)
    with open(replay_file, "wb") as f:
        pickle.dump(list(replay.buffer), f)
    final_state = {"episode": config["EPISODES"], "last_ckpt": final_ckpt}
    with open(train_state_file, "w") as f:
        json.dump(final_state, f)

    print("Training done, final saved at", final_ckpt)


if __name__ == "__main__":
    main()
