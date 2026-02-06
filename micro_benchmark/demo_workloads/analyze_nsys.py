import pandas as pd
import sys
import os


def analyze_nsys_csv(file_path):
    print(f"正在加载 Nsys CSV 文件: {file_path} ...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # --- 1. 列名适配 (处理不同版本的 nsys 列名差异) ---
    # 目标找到: 名字列(Name) 和 时长列(Duration)
    col_map = {c: c for c in df.columns}

    # 查找 Kernel 名称列
    name_col = None
    possible_names = ["Name", "Kernel Name", "Text"]
    for candidate in possible_names:
        if candidate in df.columns:
            name_col = candidate
            break

    # 查找时长列
    dur_col = None
    # nsys csv 通常是 'Duration (ns)' 或 'Duration'
    for c in df.columns:
        if "Duration" in c:
            dur_col = c
            break

    if not name_col or not dur_col:
        print("错误: 无法识别 'Name' 或 'Duration' 列。")
        print(f"当前CSV列名: {list(df.columns)}")
        print("请确认你分析的是 *_cuda_gpu_trace.csv 文件。")
        return

    print(f"使用列名 -> 名称: '{name_col}', 时长: '{dur_col}'")

    # --- 2. 数据清洗与单位转换 ---
    # Nsys 导出的 CSV 时长通常是【纳秒 (ns)】
    # 我们需要转换为【微秒 (us)】以匹配 Chrome Trace 的格式

    # 过滤无效数据 (时长 <= 0)
    df = df[df[dur_col] > 0].copy()

    # 转换: ns -> us (除以 1000)
    df["duration_us"] = df[dur_col] / 1000.0

    # --- 3. 聚合统计 ---
    stats = df.groupby(name_col)["duration_us"].agg(["count", "mean", "sum"])

    # --- 4. 格式化和排序 (保持格式一致) ---
    stats.columns = ["Count", "Avg Duration (us)", "Total Duration (us)"]

    # 保留两位小数
    stats["Avg Duration (us)"] = stats["Avg Duration (us)"].round(2)
    stats["Total Duration (us)"] = stats["Total Duration (us)"].round(2)

    # 按总时长降序排列
    stats = stats.sort_values(by="Total Duration (us)", ascending=False)

    # --- 5. 输出结果 ---
    print("\n=== Top 20 Kernels by Total Duration ===")
    print(stats.head(20))

    # 导出到 CSV
    # 去掉原扩展名，加上 _stats.csv
    base_name = os.path.splitext(file_path)[0]
    output_csv = base_name + "_stats.csv"
    stats.to_csv(output_csv)
    print(f"\n完整统计已保存至: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_nsys.py <nsys_gpu_trace.csv>")
        print("Tip: 通常分析名为 '*_cuda_gpu_trace.csv' 的文件")
    else:
        analyze_nsys_csv(sys.argv[1])
