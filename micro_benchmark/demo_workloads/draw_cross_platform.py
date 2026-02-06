import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 如果没装 seaborn 可以注释掉，只用 matplotlib 也可以


def plot_kernel_scatter():
    # ================= 配置区域 =================
    # 请在这里填入你的4个csv文件路径
    # 格式: "图例显示的名称": "文件名.csv"
    files = {
        "PyTorch - Llama3.2-1B": "torch_llama.json_stats.csv",
        "PyTorch - Resnet-50": "torch_resnet.json_stats.csv",
        "Jax - Llama3.2-1B": "jax_llama_cuda_gpu_trace_stats.csv",
        "Jax - Resnet-50": "jax_resnet.json_stats.csv",
    }

    # 定义4种不同的标记 (Marker) 和颜色
    # o=圆圈, s=方块, ^ =三角形, D=菱形
    markers = ["o", "s", "o", "s"]
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]  # 蓝, 橙, 绿, 红

    # 是否使用对数坐标轴 (推荐 True，因为 Kernel 数据的跨度通常很大)
    USE_LOG_SCALE = True
    # ===========================================

    # 设置绘图风格
    plt.style.use("seaborn-v0_8-whitegrid")  # 如果报错，可以改为 'ggplot' 或注释掉
    fig, ax = plt.subplots(figsize=(5, 4))

    # 遍历文件并绘图
    for i, (label, filepath) in enumerate(files.items()):
        try:
            # 读取数据
            df = pd.read_csv(filepath)

            # 确保列名匹配 (根据之前的脚本生成的列名)
            # x轴: Count, y轴: Avg Duration (us)
            if "Count" not in df.columns or "Avg Duration (us)" not in df.columns:
                print(f"Warning: 文件 {filepath} 列名不匹配，跳过。")
                continue

            # 提取前20行 (假设文件里已经是 Top 20，如果不是，这里可以加 .head(20))
            df_top = df.head(20)

            x = df_top["Count"]
            y = df_top["Avg Duration (us)"]

            # 绘制散点
            ax.scatter(
                x,
                y,
                label=label,
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                s=100,  # 点的大小
                alpha=0.7,  # 透明度，防止重叠
                edgecolors="w",  # 点的白边，增加对比度
                linewidth=0.5,
            )

        except FileNotFoundError:
            print(f"Error: 找不到文件 {filepath}")
        except Exception as e:
            print(f"Error 处理 {filepath}: {e}")

    # 设置坐标轴标签
    ax.set_xlabel("Call Count (Count)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Duration (us)", fontsize=12, fontweight="bold")
    # ax.set_title("Top 20 Kernels Distribution: Count vs Duration", fontsize=14)

    # 设置对数坐标
    if USE_LOG_SCALE:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
    else:
        ax.grid(True, alpha=0.3)

    # 添加图例
    ax.legend(
        title="Workloads", title_fontsize=10, fontsize=10, loc="best", frameon=True
    )

    # 调整布局并保存
    plt.tight_layout()
    output_img = "kernel_scatter_plot.pdf"
    plt.savefig(output_img, dpi=300)
    print(f"图表已保存为: {output_img}")
    plt.show()


if __name__ == "__main__":
    plot_kernel_scatter()
