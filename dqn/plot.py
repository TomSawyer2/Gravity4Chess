import re
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """
    从日志文件中解析出 [EP x] 相关数据，返回包含 ep, loss, policy_loss, value_loss 的列表。
    只有同时包含 Buffer=..., T=..., Loss=..., Policy Loss=..., Value Loss=... 的行才会被捕获。
    """
    # 编译用于匹配的正则表达式
    pattern = re.compile(
        r"\[EP\s+(\d+)\].*?"            # 匹配 [EP x] 并捕获EP号
        r"Buffer=(\d+),\s*"            # 匹配 Buffer=xxxx
        r"T=([\d\.]+),\s*"             # 匹配 T=xx.xx
        r"Loss=([\d\.]+),\s*"          # 匹配 Loss=xx.xx
        r"Policy\s+Loss=([\d\.]+),\s*" # 匹配 Policy Loss=xx.xx
        r"Value\s+Loss=([\d\.]+)"      # 匹配 Value Loss=xx.xx
    )

    episodes = []
    losses = []
    policy_losses = []
    value_losses = []

    # 逐行读取日志文件，匹配并提取所需数值
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep = int(match.group(1))
                loss = float(match.group(4))
                policy_loss = float(match.group(5))
                value_loss = float(match.group(6))

                episodes.append(ep)
                losses.append(loss)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

    return episodes, losses, policy_losses, value_losses

def plot_and_save(episodes, losses, policy_losses, value_losses, out_png="merged_plot.png"):
    """
    绘制三张子图（Loss, Policy Loss, Value Loss），并将它们合并为一个 PNG 文件。
    图中仅显示平行于 X 轴的虚线。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=680)

    def add_horizontal_lines(ax):
        """仅添加平行于 X 轴的虚线"""
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5)  # 启用水平虚线
        ax.xaxis.grid(False)  # 禁用竖线

    # Loss
    axes[0].plot(episodes, losses, label="Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("EP")
    axes[0].set_ylabel("Loss")
    add_horizontal_lines(axes[0])

    # Policy Loss
    axes[1].plot(episodes, policy_losses, color='orange', label="Policy Loss")
    axes[1].set_title("Policy Loss")
    axes[1].set_xlabel("EP")
    axes[1].set_ylabel("Policy Loss")
    add_horizontal_lines(axes[1])

    # Value Loss
    axes[2].plot(episodes, value_losses, color='green', label="Value Loss")
    axes[2].set_title("Value Loss")
    axes[2].set_xlabel("EP")
    axes[2].set_ylabel("Value Loss")
    add_horizontal_lines(axes[2])

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"图像已保存至: {out_png}")

if __name__ == "__main__":
    # 1. 读取并解析日志文件
    log_file = "output.log"  # 替换为实际的日志文件路径
    ep_list, loss_list, policy_loss_list, value_loss_list = parse_log_file(log_file)

    if len(ep_list) == 0:
        print("未在日志中解析到任何符合格式的行，请检查正则或日志格式。")
    else:
        # 2. 绘制并输出图片
        output_png = "merged_plot.png"
        plot_and_save(ep_list, loss_list, policy_loss_list, value_loss_list, out_png=output_png)
