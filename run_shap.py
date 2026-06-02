#!/usr/bin/env python
"""运行 SHAP 分析并将输出写入日志文件"""
import sys, io, os, time

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "shap_analysis_log.txt")

# 设置 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

print(f"SHAP 分析开始: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"日志同时写入: {LOG}", flush=True)

print("导入模块...", flush=True)
t_import_start = time.time()
from src.shap_analysis import main
print(f"  导入完成 ({time.time() - t_import_start:.1f}s)", flush=True)

t0 = time.time()
print("运行 main()...", flush=True)
main()
t1 = time.time()
elapsed = (t1 - t0) / 60
print(f"\n总耗时: {elapsed:.1f} 分钟", flush=True)
print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
