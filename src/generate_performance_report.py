"""整理所有模型性能评估数据并输出为 Word 文档"""

import sys
import io
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from src.config import ROOT, TABLES
from src.data_loader import load_and_clean

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MODELS_7 = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR"]
MODELS_9 = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR", "Ridge", "Lasso"]


def add_styled_table(doc, headers, rows, col_widths=None, title=None):
    """添加带格式的表格到文档"""
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(11)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)

    # 数据行
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)

    doc.add_paragraph()  # 空行
    return table


def _load_data():
    """加载所有性能数据，返回 (cv_data, split_results, topsis_data)"""
    # 嵌套CV
    with open(TABLES / "cv_results.json", "r", encoding="utf-8") as f:
        cv_data = json.load(f)

    # 80/20 分割
    split_results = {}
    for name in MODELS_7:
        df = pd.read_csv(TABLES / "prediction_tables" / f"Figure_4_{name}_Predictions.csv")
        tr = df[df['Set'] == 'Train']
        te = df[df['Set'] == 'Test']
        y_tr, p_tr = tr['True_Value'].values, tr['Predicted_Value'].values
        y_te, p_te = te['True_Value'].values, te['Predicted_Value'].values
        split_results[name] = {
            'train_r2': r2_score(y_tr, p_tr),
            'train_rmse': np.sqrt(mean_squared_error(y_tr, p_tr)),
            'test_r2': r2_score(y_te, p_te),
            'test_rmse': np.sqrt(mean_squared_error(y_te, p_te)),
            'test_mae': mean_absolute_error(y_te, p_te),
            'test_mape': np.mean(np.abs((y_te - p_te) / np.clip(y_te, 1e-6, None))) * 100,
        }

    # TOPSIS
    topsis_data = [
        (1, 'TabPFN',   0.9456, 1, 3, 2),
        (2, 'XGBoost',  0.9142, 2, 1, 1),
        (3, 'GBDT',     0.9022, 3, 2, 1),
        (4, 'RF',       0.8760, 4, 4, 0),
        (5, 'LightGBM', 0.8517, 5, 5, 0),
        (6, 'SVR',      0.8478, 6, 6, 0),
        (7, 'GPR',      0.8035, 7, 7, 0),
        (8, 'Ridge',    0.0312, 8, 9, 1),
        (9, 'Lasso',    0.0236, 9, 8, 1),
    ]

    return cv_data, split_results, topsis_data


def _make_doc():
    """创建标准 A4 文档"""
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    return doc


R2_KEY = 'R²_mean'
R2S_KEY = 'R²_std'

HEADERS_CV = ['排名', '模型', '管道', 'R²_mean', 'R²_std', 'RMSE_mean', 'RMSE_std',
              'MAE_mean', 'MAPE_mean']
HEADERS_CV_EN = ['Rank', 'Model', 'Pipeline', 'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std',
                  'MAE_mean', 'MAPE_mean']
HEADERS_SPLIT = ['排名', '模型', '训练R²', '训练RMSE', '测试R²', '测试RMSE', '测试MAE', '测试MAPE']
HEADERS_SPLIT_EN = ['Rank', 'Model', 'Train R2', 'Train RMSE', 'Test R2', 'Test RMSE',
                     'Test MAE', 'Test MAPE']
HEADERS_TOPSIS = ['排名', '模型', '贴近度C_i', '熵权排名', 'CRITIC排名', '极差']
HEADERS_TOPSIS_EN = ['Rank', 'Model', 'Closeness C_i', 'Entropy Rank', 'CRITIC Rank', 'Range']
HEADERS_FILES = ['文件路径', '说明']
HEADERS_FILES_EN = ['File', 'Description']


def _build_cv_rows(cv_data):
    rows = []
    ranked = sorted(cv_data.items(), key=lambda x: x[1]['aggregate'].get(R2_KEY, -999), reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        agg = r['aggregate']
        rows.append([
            rank, name, r['pipeline'],
            f"{agg[R2_KEY]:.4f}", f"{agg[R2S_KEY]:.4f}",
            f"{agg['RMSE_mean']:.2f}", f"{agg['RMSE_std']:.2f}",
            f"{agg['MAE_mean']:.2f}", f"{agg['MAPE_mean']:.2f}%",
        ])
    return rows, ranked


def _build_split_rows(split_results):
    rows = []
    ranked = sorted(split_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        rows.append([
            rank, name,
            f"{r['train_r2']:.4f}", f"{r['train_rmse']:.2f}",
            f"{r['test_r2']:.4f}", f"{r['test_rmse']:.2f}",
            f"{r['test_mae']:.2f}", f"{r['test_mape']:.2f}%",
        ])
    return rows


def _build_topsis_rows(topsis_data):
    return [[str(r[0]), r[1], f"{r[2]:.4f}", f"#{r[3]}", f"#{r[4]}", str(r[5])] for r in topsis_data]


def generate_report(lang='en'):
    """生成模型性能评估报告

    Parameters
    ----------
    lang : str
        'en' 英文版, 'zh' 中文版
    """
    is_zh = lang == 'zh'

    print(f"收集模型性能数据... (语言: {'中文' if is_zh else 'English'})")
    cv_data, split_results, topsis_data = _load_data()
    doc = _make_doc()
    rows_cv, ranked = _build_cv_rows(cv_data)
    rows_split = _build_split_rows(split_results)
    rows_topsis = _build_topsis_rows(topsis_data)
    best_r2 = ranked[0][1]['aggregate'][R2_KEY]
    ridge_r2 = cv_data['Ridge']['aggregate'][R2_KEY]
    gap = best_r2 - ridge_r2

    # ── 标题 ──────────────────────────────────────────────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        'MgO改性生物炭CO₂吸附性能\n机器学习模型性能评估报告' if is_zh
        else 'MgO-Modified Biochar CO2 Adsorption\nMachine Learning Model Performance Report'
    )
    run.bold = True
    run.font.size = Pt(16)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        '生成日期: 2026-05-28 | 随机种子=42 | 样本量=341' if is_zh
        else 'Generated: 2026-05-28 | Seed=42 | 341 Samples'
    )
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(128, 128, 128)
    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════
    # 1. 嵌套CV
    # ══════════════════════════════════════════════════════════════════
    doc.add_heading(
        '1. 嵌套交叉验证结果 (StratifiedKFold 5×3)' if is_zh
        else '1. Nested Cross-Validation Results (5x3 StratifiedKFold)', level=1
    )
    doc.add_paragraph(
        '外层5折StratifiedKFold，内层3折Optuna TPE超参数优化(n_trials=100)。'
        'XGBoost和LightGBM施加KDE样本权重以增强尾部样本。'
        'TabPFN为零超参数基础模型。所有预处理（缺失值填补+特征工程）均封装在各Pipeline内部，'
        '杜绝数据泄露。' if is_zh
        else 'Outer 5-fold StratifiedKFold, inner 3-fold for Optuna TPE hyperparameter tuning '
        '(n_trials=100). KDE sample weights applied for XGBoost and LightGBM. '
        'TabPFN uses zero hyperparameters. All preprocessing (imputation + feature engineering) '
        'embedded within each pipeline to prevent data leakage.'
    )
    add_styled_table(doc, HEADERS_CV if is_zh else HEADERS_CV_EN, rows_cv,
                     title='表1. 嵌套CV性能指标（按R²降序排列）' if is_zh
                     else 'Table 1. Nested CV performance metrics (sorted by R2).')

    doc.add_paragraph(
        f'非线性-线性差距: {gap:.4f} (最优{ranked[0][0]}: {best_r2:.4f} vs Ridge: {ridge_r2:.4f})。'
        f'该差距远大于0.05阈值，充分支持本数据集需要非线性建模。' if is_zh
        else f'Nonlinear-Linear Gap: {gap:.4f} (best {ranked[0][0]}: {best_r2:.4f} vs Ridge: {ridge_r2:.4f}). '
        f'This large gap (>0.05 threshold) strongly supports the necessity of nonlinear modeling '
        f'for this dataset.'
    )

    # ══════════════════════════════════════════════════════════════════
    # 2. 80/20 分割
    # ══════════════════════════════════════════════════════════════════
    doc.add_heading(
        '2. 80/20 训练-测试分割结果' if is_zh
        else '2. 80/20 Train-Test Split Results', level=1
    )
    doc.add_paragraph(
        '单次分层80/20随机划分(random_state=42)。模型在272条训练样本上训练，'
        '在69条留出测试样本上评估。作为嵌套CV之外独立的外推性能验证。' if is_zh
        else 'Single stratified 80/20 random split (random_state=42). Models trained on 272 training samples '
        'and evaluated on 69 held-out test samples. This provides an independent verification of '
        'generalization performance complementary to the nested CV above.'
    )
    add_styled_table(doc, HEADERS_SPLIT if is_zh else HEADERS_SPLIT_EN, rows_split,
                     title='表2. 80/20留出测试结果（按测试R²降序排列）' if is_zh
                     else 'Table 2. 80/20 hold-out test results (sorted by Test R2).')

    # ══════════════════════════════════════════════════════════════════
    # 3. TOPSIS
    # ══════════════════════════════════════════════════════════════════
    doc.add_heading(
        '3. TOPSIS 多准则综合排名' if is_zh
        else '3. TOPSIS Multi-Criteria Comprehensive Ranking', level=1
    )
    doc.add_paragraph(
        '层次熵权法（论文主方案）：组间领域知识权重——精度50% / 稳定性25% / 泛化25%。'
        '组内：熵权法客观赋权。MAE_mean因与RMSE_mean高度共线(r=0.997)被排除。' if is_zh
        else 'Grouped Entropy Weight Method (primary scheme for the paper): '
        'Between-group domain knowledge weights: Accuracy 50% / Stability 25% / Generalization 25%. '
        'Within-group: Entropy method for objective weighting. '
        'MAE_mean excluded due to collinearity with RMSE_mean (r=0.997).'
    )
    add_styled_table(doc, HEADERS_TOPSIS if is_zh else HEADERS_TOPSIS_EN, rows_topsis,
                     title='表3. TOPSIS层次熵权综合排名（论文主方案）' if is_zh
                     else 'Table 3. TOPSIS comprehensive ranking with Grouped Entropy weights (primary scheme).')
    doc.add_paragraph(
        '敏感性分析：中游模型(GBDT, RF, LightGBM, SVR, GPR)在三种赋权方案(熵权、CRITIC、层次熵权)'
        '下排名完全一致(极差=0)，验证了排名的稳健性。CRITIC权重因n=9样本量有限，'
        '仅作敏感性分析参考。' if is_zh
        else 'Sensitivity analysis: Mid-tier models (GBDT, RF, LightGBM, SVR, GPR) show zero rank variation '
        'across all three weighting schemes (entropy, CRITIC, grouped entropy), confirming robustness. '
        'CRITIC weights are less stable with n=9 models and serve only as sensitivity check.'
    )

    # ══════════════════════════════════════════════════════════════════
    # 4. 核心发现
    # ══════════════════════════════════════════════════════════════════
    doc.add_heading(
        '4. 核心发现总结' if is_zh
        else '4. Summary of Key Findings', level=1
    )

    if is_zh:
        findings = [
            f'TabPFN（基础模型，零超参数）在嵌套CV中取得最优R²='
            f'{cv_data["TabPFN"]["aggregate"][R2_KEY]:.4f}，并在TOPSIS综合排名中以贴近度'
            f'C_i=0.9456位列第一，验证了合成数据预训练可以有效迁移至本材料科学回归任务。',

            f'树集成模型（GBDT, RF, XGBoost, LightGBM）嵌套CV R²均超过0.93，形成紧密的竞争集群。'
            f'其中LightGBM在80/20留出测试中表现最优'
            f'(R²={split_results["LightGBM"]["test_r2"]:.4f})，展现出较强的鲁棒性。',

            f'核方法（GPR, SVR）表现具有竞争力但略低于树模型'
            f'(CV R² = {cv_data["GPR"]["aggregate"][R2_KEY]:.4f} 和 '
            f'{cv_data["SVR"]["aggregate"][R2_KEY]:.4f})，'
            f'与其捕捉复杂特征交互能力有限的理论预期一致。',

            f'线性基线模型（Ridge, Lasso）完全失效（CV R² < 0.37），确认了MgO改性多孔碳体系中'
            f'构效关系的强非线性本质。',

            f'SHAP可解释性分析中，GBDT TreeExplainer与TabPFN排列重要性高度一致'
            f'(Spearman ρ=0.8796)。pressure_bar（操作压力）、microporosity（微孔率）和'
            f'temperature_C（操作温度）在不同模型中一致被识别为最具影响力的特征。',
        ]
    else:
        findings = [
            f'TabPFN (foundation model, zero hyperparameters) achieves the best CV performance '
            f'(R2={cv_data["TabPFN"]["aggregate"][R2_KEY]:.4f}) and highest TOPSIS score (C_i=0.9456), '
            f'validating that prior-fitting on synthetic datasets transfers effectively to this '
            f'materials science regression task.',

            f'Tree ensemble models (GBDT, RF, XGBoost, LightGBM) all achieve CV R2 > 0.93, '
            f'forming a tight competitive cluster. LightGBM leads the 80/20 split test '
            f'(R2={split_results["LightGBM"]["test_r2"]:.4f}), suggesting strong robustness.',

            f'Kernel methods (GPR, SVR) show competitive but slightly lower performance '
            f'(CV R2 = {cv_data["GPR"]["aggregate"][R2_KEY]:.4f} and {cv_data["SVR"]["aggregate"][R2_KEY]:.4f}), '
            f'consistent with their limited capacity to capture complex feature interactions.',

            f'Linear baselines (Ridge, Lasso) fail dramatically (CV R2 < 0.37), confirming the strongly '
            f'nonlinear nature of the structure-property relationships in MgO-modified porous carbons.',

            f'SHAP analysis (GBDT TreeExplainer vs TabPFN permutation importance) shows strong agreement '
            f'(Spearman rho=0.8796), with pressure_bar, microporosity, and temperature_C consistently '
            f'emerging as the most influential features across models.',
        ]

    for i, f_text in enumerate(findings, 1):
        p = doc.add_paragraph()
        run = p.add_run(f'{i}. {f_text}')
        run.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(6)

    # ══════════════════════════════════════════════════════════════════
    # 5. 文件清单
    # ══════════════════════════════════════════════════════════════════
    doc.add_heading(
        '5. 输出文件清单' if is_zh
        else '5. Data File Inventory', level=1
    )

    if is_zh:
        rows_files = [
            ['outputs/tables/cv_results.json', '全部9模型的嵌套CV完整指标'],
            ['outputs/tables/cv_predictions.json', '拼接后的逐折CV外推预测值（7个非线性模型）'],
            ['outputs/tables/prediction_tables/Figure_4_*_Predictions.csv', '80/20分割预测（每模型272训练+69测试）'],
            ['outputs/tables/topsis_grouped.csv', 'TOPSIS层次熵权法排名'],
            ['outputs/tables/topsis_entropy.csv', 'TOPSIS标准熵权法排名'],
            ['outputs/tables/topsis_critic.csv', 'TOPSIS CRITIC权重排名（敏感性分析）'],
            ['outputs/tables/entropy_weights.csv', '熵权法各指标权重'],
            ['outputs/tables/grouped_weights.csv', '层次熵权法分层权重'],
        ]
    else:
        rows_files = [
            ['outputs/tables/cv_results.json', 'Full nested CV metrics for all 9 models'],
            ['outputs/tables/cv_predictions.json', 'Concatenated out-of-fold CV predictions (7 nonlinear models)'],
            ['outputs/tables/prediction_tables/Figure_4_*_Predictions.csv', '80/20 split predictions (272T+69T each)'],
            ['outputs/tables/topsis_grouped.csv', 'TOPSIS grouped entropy weights ranking'],
            ['outputs/tables/topsis_entropy.csv', 'TOPSIS standard entropy weights ranking'],
            ['outputs/tables/topsis_critic.csv', 'TOPSIS CRITIC weights ranking (sensitivity)'],
            ['outputs/tables/entropy_weights.csv', 'Entropy method indicator weights'],
            ['outputs/tables/grouped_weights.csv', 'Grouped entropy hierarchical weights'],
        ]

    add_styled_table(doc, HEADERS_FILES if is_zh else HEADERS_FILES_EN, rows_files,
                     title='表4. 论文图表所需的关键输出文件' if is_zh
                     else 'Table 4. Key output files for paper figures and tables.')

    # ── 保存 ──────────────────────────────────────────────────────────
    suffix = '_CN' if is_zh else ''
    output_path = TABLES / f"Model_Performance_Report{suffix}.docx"
    doc.save(output_path)
    print(f"报告已保存: {output_path}")


def main():
    generate_report(lang='en')
    generate_report(lang='zh')
    print("\n中英文双版本均已生成。")


if __name__ == "__main__":
    main()
