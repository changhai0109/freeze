import json
import pandas as pd
import sys


def analyze_trace(file_path):
    print(f"正在加载 trace 文件: {file_path} ...")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # Chrome Trace 有时是一个列表，有时是一个包含 'traceEvents' 的字典
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    # 转换为 DataFrame
    df = pd.DataFrame(events)

    # 1. 过滤有效事件
    # ph='X' 表示 Complete Event (有开始和持续时间)
    # dur 存在且大于 0
    if "dur" not in df.columns:
        print("错误: Trace 中没有找到 'dur' (duration) 字段。")
        return

    valid_events = df[(df["ph"] == "X") & (df["dur"] > 0)].copy()

    # 2. (可选) 过滤 GPU Kernel
    # 不同的 Profiler (JAX, PyTorch, Nsys) 的 category (cat) 命名不同。
    # 如果你想统计所有事件，可以跳过这一步。
    # 这里是一个简单的关键词过滤，你可以先 print(valid_events['cat'].unique()) 看看有哪些类别
    # valid_events = valid_events[valid_events['cat'].astype(str).str.contains('kernel|cuda|gpu', case=False, na=False)]

    if valid_events.empty:
        print("没有找到符合条件的事件。")
        return

    # 3. 聚合统计
    # 按名称分组，计算数量、平均时长(us)、总时长(us)
    stats = valid_events.groupby("name")["dur"].agg(["count", "mean", "sum"])

    # 4. 格式化和排序
    stats.columns = ["Count", "Avg Duration (us)", "Total Duration (us)"]
    stats["Avg Duration (us)"] = stats["Avg Duration (us)"].round(2)

    # 按总时长降序排列
    stats = stats.sort_values(by="Total Duration (us)", ascending=False)

    # 5. 输出结果
    print("\n=== Top 20 Kernels by Total Duration ===")
    print(stats.head(20))

    # 导出到 CSV
    output_csv = file_path + "_stats.csv"
    stats.to_csv(output_csv)
    print(f"\n完整统计已保存至: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <trace_file.json>")
    else:
        analyze_trace(sys.argv[1])
