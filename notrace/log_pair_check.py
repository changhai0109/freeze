#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import Counter, namedtuple

# 日志格式假设：
# nvbit_at_cuda_event: ENTER cuDevicePrimaryCtxRetain
# nvbit_at_cuda_event: EXIT cuCtxGetCurrent
# CUDA API ENTER: cuCtxGetCurrent
# CUDA API EXIT: cuCtxGetDevice

NVBIT_PATTERN = re.compile(
    r'^nvbit_at_cuda_event:\s+(ENTER|EXIT)\s+(\S+)'
)

CUDA_API_PATTERN = re.compile(
    r'^CUDA API\s+(ENTER|EXIT):\s+(\S+)'
)

Entry = namedtuple('Entry', ['event', 'func', 'line_no', 'raw'])


def parse_log(filename):
    nvbit_entries = []
    cuda_entries = []

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            text = line.rstrip('\n')

            m1 = NVBIT_PATTERN.match(text)
            if m1:
                evt, func = m1.groups()
                nvbit_entries.append(Entry(evt, func, i, text))
                continue

            m2 = CUDA_API_PATTERN.match(text)
            if m2:
                evt, func = m2.groups()
                cuda_entries.append(Entry(evt, func, i, text))
                continue

    return nvbit_entries, cuda_entries


def build_counters(entries):
    """
    entries: list[Entry]
    返回 Counter((event, func) -> count)
    """
    c = Counter()
    for e in entries:
        c[(e.event, e.func)] += 1
    return c


def print_counters(nvbit_counter, cuda_counter):
    print("==== 计数统计 (按 (ENTER/EXIT, 函数名) 统计) ====\n")

    print("nvbit_at_cuda_event 计数：")
    for k, v in sorted(nvbit_counter.items()):
        print(f"  {k}: {v}")

    print("\nCUDA API 计数：")
    for k, v in sorted(cuda_counter.items()):
        print(f"  {k}: {v}")

    if nvbit_counter == cuda_counter:
        print("\n[计数检查] 通过：")
        print("  nvbit_at_cuda_event 和 CUDA API 在 (ENTER/EXIT, 函数名) 上完全一致。")
    else:
        print("\n[计数检查] 失败：nvbit 与 CUDA API 计数不一致，差异如下：")
        all_keys = set(nvbit_counter.keys()) | set(cuda_counter.keys())
        for key in sorted(all_keys):
            n = nvbit_counter.get(key, 0)
            c = cuda_counter.get(key, 0)
            if n != c:
                print(f"  {key}: nvbit={n}, cuda={c}")


def check_enter_exit_pairs(entries, label):
    """
    检查某一类记录内部的 ENTER/EXIT 是否成对、嵌套是否正常。
    返回错误信息列表（字符串）。
    """
    stack = []
    errors = []

    for e in entries:
        if e.event == 'ENTER':
            stack.append(e)
        elif e.event == 'EXIT':
            if not stack:
                errors.append(
                    f"[{label}] line {e.line_no}: EXIT {e.func} 没有匹配的 ENTER\n"
                    f"    {e.raw}"
                )
                continue

            last = stack.pop()
            if last.func != e.func:
                errors.append(
                    f"[{label}] line {e.line_no}: EXIT {e.func} 与上一个 ENTER 不匹配\n"
                    f"    ENTER at line {last.line_no}: {last.raw}\n"
                    f"    EXIT  at line {e.line_no}: {e.raw}"
                )

    # 栈里剩下的都是没有 EXIT 的 ENTER
    for e in stack:
        errors.append(
            f"[{label}] line {e.line_no}: ENTER {e.func} 没有匹配的 EXIT\n"
            f"    {e.raw}"
        )

    return errors


def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} path/to/log.txt")
        sys.exit(1)

    log_file = sys.argv[1]
    nvbit_entries, cuda_entries = parse_log(log_file)

    print(f"解析到 nvbit_at_cuda_event 条数: {len(nvbit_entries)}")
    print(f"解析到 CUDA API 条数:         {len(cuda_entries)}\n")

    # 1) 计数检查
    nvbit_counter = build_counters(nvbit_entries)
    cuda_counter = build_counters(cuda_entries)
    print_counters(nvbit_counter, cuda_counter)

    # 2) ENTER/EXIT 栈一致性检查
    print("\n==== ENTER/EXIT 配对与嵌套检查 ====\n")

    errors_nvbit = check_enter_exit_pairs(nvbit_entries, "nvbit")
    errors_cuda = check_enter_exit_pairs(cuda_entries, "cuda")

    if not errors_nvbit:
        print("[nvbit] ENTER/EXIT 检查通过：内部配对且嵌套合法。")
    else:
        print("[nvbit] ENTER/EXIT 检查发现问题：")
        for e in errors_nvbit:
            print(" -", e)

    print()

    if not errors_cuda:
        print("[cuda ] ENTER/EXIT 检查通过：内部配对且嵌套合法。")
    else:
        print("[cuda ] ENTER/EXIT 检查发现问题：")
        for e in errors_cuda:
            print(" -", e)


if __name__ == "__main__":
    main()
    print("Remember: disable api_call_flaggers if you see unmatched numbers, because they bypass nvbit_at_cuda_event hooks sometimes")
