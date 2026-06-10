#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正式全量重跑 — 挂机模式

参数: n_trials=100, CV=5×3 StratifiedKFold, SHAP n_repeats=30
输出: outputs/ (覆盖旧版本)
"""

import sys
import io
import os
import time
import json
import warnings
import traceback
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
LOG_DIR = OUTPUTS / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)

def run_step(name: str, module: str, args: list[str] = None) -> int:
    """运行一个 Python 模块并记录日志。返回退出码。"""
    log(f"{'='*60}")
    log(f"开始: {name}")
    log(f"{'='*60}")

    log_path = LOG_DIR / f"{name.replace(' ', '_').replace('/', '_')}.log"
    t0 = time.time()

    import subprocess
    cmd = [sys.executable, "-m"] + module.split() if " " not in module else [sys.executable] + module.split()
    if args:
        cmd.extend(args)

    # For scripts that are not modules (plot_80_20_marginal.py etc.)
    if not module.startswith("src."):
        cmd = [sys.executable, str(ROOT / module)]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=28800,  # 8 hour max per step
        )
        elapsed = time.time() - t0

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

        if result.returncode == 0:
            log(f"✓ {name} 完成 ({elapsed/60:.1f} min)")
        else:
            log(f"✗ {name} 失败 (exit={result.returncode}, {elapsed/60:.1f} min)")
            log(f"  日志: {log_path}")
            # Print last 20 lines of stderr
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-10:]:
                log(f"  stderr: {line}")

        return result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f"✗ {name} 超时 ({elapsed/60:.1f} min > 8h)")
        return 1
    except Exception as e:
        elapsed = time.time() - t0
        log(f"✗ {name} 异常: {e} ({elapsed/60:.1f} min)")
        return 1


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()
    log("=" * 70)
    log(" 正式全量重跑")
    log(f" 参数: n_trials=100, CV=5×3, SHAP n_repeats=30")
    log(f" 输出: {OUTPUTS}")
    log(f" 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    failures = []
    steps = [
        # ── 阶段 1: 模型训练 ──
        ("01_Train_CV",             "src.train",            []),
        # ── 阶段 2: 评估与排名 ──
        ("02_TOPSIS",               "src.topsis",           []),
        ("03_80_20_Export",         "src.export_prediction_tables", []),
        # ── 阶段 3: SHAP 可解释性 ──
        ("04_SHAP_Analysis",        "src.shap_analysis",    []),
        # ── 阶段 4: 论文图表 ──
        ("05_Figure_S1",            "src.plotting",         ["--figure", "S1"]),
        ("06_Figure_1",             "src.plotting",         ["--figure", "1"]),
        ("07_Figure_2",             "src.plotting",         ["--figure", "2"]),
        ("08_Figure_3",             "src.plotting",         ["--figure", "3"]),
        ("09_Figure_4_TabPFN",      "src.plotting",         ["--figure", "4"]),
        ("10_Figure_5",             "src.plotting",         ["--figure", "5"]),
        ("11_Figure_6",             "src.plotting",         ["--figure", "6"]),
        ("12_Figure_7",             "src.plotting",         ["--figure", "7"]),
        ("13_Figure_8",             "src.plotting",         ["--figure", "8"]),
        ("14_SHAP_Bar",             "src.plotting",         ["--figure", "BAR"]),
        ("15_Figure_4_7Models",     "plot_80_20_marginal.py", []),
        # ── 阶段 5: 报告 ──
        ("16_Generate_Report",      "src.generate_report",           []),
        ("17_Performance_Report",   "src.generate_performance_report", []),
    ]

    for name, module, args in steps:
        rc = run_step(name, module, args)
        if rc != 0:
            failures.append(name)
            # 关键步骤失败则终止
            if name in ("01_Train_CV", "04_SHAP_Analysis"):
                log(f"✗ 关键步骤 {name} 失败，终止流水线")
                break

    # ══════════════════════════════════════════════════════════════════════════
    # 最终汇总
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start
    log("")
    log("=" * 70)
    log(f" 全量重跑结束 ({total_elapsed/3600:.1f} h)")
    log(f" 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # 产物统计
    figures = sorted((OUTPUTS / "figures").rglob("*.png")) if (OUTPUTS / "figures").exists() else []
    tables = sorted((OUTPUTS / "tables").glob("*")) if (OUTPUTS / "tables").exists() else []
    models = sorted((OUTPUTS / "models").glob("*.pkl")) if (OUTPUTS / "models").exists() else []

    log(f"\n  Figures: {len(figures)}")
    for f in figures:
        log(f"    {f.name}")
    log(f"\n  Tables: {len(tables)}")
    log(f"\n  Models: {len(models)}")
    for f in models:
        log(f"    {f.name}")

    if failures:
        log(f"\n  ✗ {len(failures)} 个步骤失败: {failures}")
        return 1
    else:
        log(f"\n  ✓ 全部 {len(steps)} 个步骤通过")
        return 0


if __name__ == "__main__":
    sys.exit(main())
