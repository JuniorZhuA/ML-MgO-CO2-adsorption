"""端到端自动化流水线入口

按依赖顺序串联所有步骤，删除 outputs/ 后一键重建全部结果。

用法:
    python -m src.main              # 完整运行（含耗时较长的嵌套CV）
    python -m src.main --skip-train  # 跳过训练，仅重新生成TOPSIS+SHAP+报告
"""

import sys
import io
import argparse
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def step(name: str):
    """打印步骤标题。"""
    print(f"\n{'#' * 70}")
    print(f"#  {name}")
    print(f"{'#' * 70}")


def run_train():
    """步骤5: 嵌套CV + 最终模型训练 → cv_results.json + 9个.pkl"""
    step("步骤5: 嵌套CV训练 (StratifiedKFold 5×3, Optuna n_trials=100)")
    t0 = time.time()
    from src.train import main as train_main
    train_main()
    print(f"  耗时: {(time.time() - t0) / 60:.1f} min")


def run_shap():
    """步骤6: SHAP可解释性 → VIF/聚类/SHAP/一致性CSV"""
    step("步骤6: SHAP可解释性分析")
    from src.shap_analysis import main as shap_main
    shap_main()


def run_topsis():
    """TOPSIS多准则决策 → 3方案6个CSV"""
    step("TOPSIS: 三方案综合排名")
    from src.topsis import main as topsis_main
    topsis_main()


def run_report():
    """生成Word综合进展报告"""
    step("生成: 项目综合进展报告")
    from src.generate_report import main as report_main
    report_main()


def main():
    parser = argparse.ArgumentParser(description="全链路自动化流水线")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="跳过嵌套CV训练（假设 outputs/models/ 和 cv_results.json 已存在）"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    for d in ["models", "tables", "figures"]:
        (OUTPUTS / d).mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Step 1: 训练 ──
    if not args.skip_train:
        run_train()
    else:
        print("[跳过] 嵌套CV训练（--skip-train）")

    # ── Step 2: SHAP + TOPSIS 可并行，顺序执行以保持输出清晰 ──
    run_shap()
    run_topsis()

    # ── Step 3: 报告 ──
    run_report()

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'#' * 70}")
    print(f"#  全链路完成！总耗时: {elapsed:.1f} min")
    print(f"{'#' * 70}")
    print(f"\n输出目录: {OUTPUTS}/")
    print(f"  models/  — 9个最终模型 (.pkl)")
    print(f"  tables/  — 全部中间结果CSV + cv_results.json")
    print(f"  figures/ — 论文图表 (.png)")
    print(f"  项目综合进展报告.docx")


if __name__ == "__main__":
    main()
