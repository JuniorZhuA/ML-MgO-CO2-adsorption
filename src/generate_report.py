"""生成项目综合进展报告 Word 文档"""
import sys, io
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"


def set_cell(cell, text, bold=False, align='center', font_size=9, font_name='Times New Roman'):
    """统一设置单元格格式"""
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = {'left': 0, 'center': 1, 'right': 2}[align]
    run = p.add_run(str(text))
    run.font.size = Pt(font_size)
    run.font.name = font_name
    run.bold = bold
    # 设置中文字体
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.makeelement(qn('w:rFonts'), {})
    rFonts.set(qn('w:eastAsia'), '宋体')
    rPr.insert(0, rFonts)


def add_styled_table(doc, headers, rows, col_widths=None):
    """添加带格式的表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for j, h in enumerate(headers):
        set_cell(table.rows[0].cells[j], h, bold=True, font_size=9)

    # 数据行
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell(table.rows[i + 1].cells[j], val, bold=(j == 0), font_size=9)

    return table


def main():
    # 仅在独立运行时设置编码；被 main.py 调用时 stdout 已配置为 utf-8
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    doc = Document()

    # 页面设置
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)

    # === 标题页 ===
    title = doc.add_heading('生物质衍生MgO改性多孔碳CO₂吸附性能\n机器学习预测研究 —— 综合进展报告', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f'生成日期：2026年5月21日', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f'项目路径：H:\\study\\machine learning\\NEW-ML-MgOBC', style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ============================================================
    # 1. 项目概述
    # ============================================================
    doc.add_heading('1. 项目概述', level=1)
    doc.add_paragraph(
        '本研究利用文献汇编的341条实验数据，构建机器学习模型预测生物质衍生MgO改性多孔碳的CO₂吸附量（mg/g）。'
        '共评估9个模型（2个线性基线 + 6个传统非线性模型 + 1个表格基础模型TabPFN），'
        '采用4条预处理管道、嵌套交叉验证（StratifiedKFold 5×3）、Optuna TPE超参调优（n_trials=100），'
        '以SHAP可解释性分析揭示特征对CO₂吸附的驱动机制。'
    )
    doc.add_paragraph(
        '论文标题：Machine Learning Prediction of CO₂ Adsorption Capacity on Biomass-Derived '
        'MgO-Modified Porous Carbons: From Traditional Models to Tabular Foundation Models with SHAP Interpretability'
    )

    # ============================================================
    # 2. 数据概况
    # ============================================================
    doc.add_heading('2. 数据概况', level=1)
    doc.add_paragraph('原始数据：0514biochar-MgOdata.xlsx，341条样本，17列（双行表头），文献汇编。')
    doc.add_paragraph('目标变量：CO₂ uptake（范围 1.32–292.67 mg/g）。')

    doc.add_heading('2.1 预处理流程', level=2)
    doc.add_paragraph(
        '• 列名规范化：赋予16个规范列名，丢弃References（文献引用）\n'
        '• 数值转换：所有数值列通过pd.to_numeric(errors="coerce")转换\n'
        '• Unicode统一：MgO_precursors中的特殊字符统一处理\n'
        '• NaN语义修正：5个工艺列（Activation1/2, Carbonization1/2, Post_treatment）的NaN替换为"none"，表示"未执行该步骤"\n'
        '• 类别合并：合并"impregnation"和"wetness impregnation"为同一类别'
    )

    doc.add_heading('2.2 缺失值处理', level=2)
    add_styled_table(doc,
        ['特征', '缺失率', '处理方法'],
        [
            ['SBET', '4.4%', 'KNNImputer (k=5)'],
            ['Vtotal', '4.4%', 'KNNImputer (k=5)'],
            ['Vmicro', '42.2%', 'IterativeImputer (BayesianRidge, max_iter=20)，物理约束Vmicro≤Vtotal'],
        ]
    )

    doc.add_heading('2.3 特征工程', level=2)
    doc.add_paragraph(
        '• 正则提取：从5个工艺文本列中正则提取温度(°C)和时长(h)数值\n'
        '• 工艺分类：构建8个分类特征（carbon_precursors, MgO_precursors, Mg_loading_method, act1_type, act2_type, carb1_type, carb2_type, post_treatment_type）\n'
        '• 5个领域复合特征：'
    )
    add_styled_table(doc,
        ['特征名', '公式', '物理意义'],
        [
            ['Vmeso', 'Vtotal − Vmicro', '介孔体积'],
            ['Microporosity', 'Vmicro / Vtotal', '微孔率，CO₂物理吸附关键指标'],
            ['MgO Surface Density', 'MgO Mass Ratio / (SBET / 1000)', '单位表面积MgO负载量'],
            ['T·ln(P)', 'Temperature × ln(Pressure + 0.01)', '吸附条件耦合项（吸附势理论）'],
            ['1/T', '1 / (Temperature + 273.15)', '温度倒数（1/K），van\'t Hoff自然变量'],
        ]
    )
    doc.add_paragraph('最终特征集：27个特征（19个数值 + 8个分类）。')

    # ============================================================
    # 3. 模型与管道
    # ============================================================
    doc.add_heading('3. 模型与预处理管道', level=1)

    doc.add_heading('3.1 9个模型', level=2)
    doc.add_paragraph(
        '线性基线（2个）：Ridge回归（L2）、Lasso回归（L1）\n'
        '树模型（4个）：Random Forest、XGBoost、LightGBM、GradientBoostingRegressor\n'
        '核方法（2个）：SVR（RBF核）、Gaussian Process Regression\n'
        '基础模型（1个）：TabPFN（Nature 2025, Hollmann et al.）— 基于Transformer的表格基础模型，零超参'
    )

    doc.add_heading('3.2 四条预处理管道', level=2)
    add_styled_table(doc,
        ['管道', '适用模型', '分类变量', '数值缩放', 'NaN处理'],
        [
            ['A', 'Ridge, Lasso', 'OneHotEncoder', 'StandardScaler', 'SimpleImputer(median)'],
            ['B', 'RF, XGBoost, LightGBM, GBDT', 'OrdinalEncoder', '无', '原生支持'],
            ['C', 'SVR, GPR', 'TargetEncoder', 'StandardScaler + log1p(y)', 'SimpleImputer(median)'],
            ['D', 'TabPFN', 'OrdinalEncoder', '无（内置预处理）', '原生支持'],
        ]
    )

    # ============================================================
    # 4. 模型性能
    # ============================================================
    doc.add_heading('4. 模型性能评估', level=1)

    doc.add_heading('4.1 单指标排名（按R²）', level=2)
    doc.add_paragraph('CV策略：StratifiedKFold 外层5折 × 内层3折，Optuna TPE n_trials=100，KDE样本权重。')

    # 读取CV结果（从TOPSIS表格获取完整指标）并按R²降序排列
    cv_df = pd.read_csv(TABLES / 'topsis_entropy.csv')
    cv_df = cv_df.sort_values('R²_mean', ascending=False).reset_index(drop=True)
    add_styled_table(doc,
        ['R²排名', '模型', 'R²_mean', 'R²_std', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAPE%', 'q10_RMSE', 'q90_RMSE'],
        [[str(cv_df['R²_mean'].rank(ascending=False).astype(int).iloc[i]), r['模型'],
          f"{r['R²_mean']:.4f}", f"{r['R²_std']:.4f}",
          f"{r['RMSE_mean']:.1f}", f"{r['RMSE_std']:.1f}",
          f"{r['MAE_mean']:.1f}", f"{r['MAPE_mean']:.1f}",
          f"{r['tail_rmse_q10_mean']:.1f}", f"{r['tail_rmse_q90_mean']:.1f}"]
         for i, (_, r) in enumerate(cv_df.iterrows())]
    )

    doc.add_paragraph(
        '关键发现：\n'
        '• 最优模型：TabPFN（R²=0.9692），零超参条件下超越所有调优模型\n'
        '• 非线性-线性R²差距：0.592（0.9692 vs 0.3608），远超0.05阈值 → 非线性建模必要性充分证明\n'
        '• TabPFN vs RF：0.9692 vs 0.9503（TabPFN胜，ΔR²=0.0189）\n'
        '• XGBoost的RMSE_std最小（1.39），预测稳定性最优'
    )

    # ============================================================
    # 5. TOPSIS
    # ============================================================
    doc.add_heading('5. TOPSIS 多准则综合排名', level=1)

    doc.add_paragraph(
        '采用TOPSIS多准则决策方法，综合8项评价指标，以三种赋权方案计算各模型相对贴近度C_i，'
        '交叉验证排名的稳健性。'
    )

    doc.add_heading('5.1 层次熵权法（Grouped Entropy）★ 论文主方案', level=2)
    doc.add_paragraph(
        '设计逻辑：标准熵权法对8个指标独立赋权，但8个指标中误差类指标（RMSE、MAE、MAPE、'
        'tail_q10、tail_q90）占5席，精度类指标仅R²占1席。即使各指标熵权接近等权，'
        '误差维度集体占据约63%的权重，构成"维度重复计数（Double Counting）"。'
    )
    doc.add_paragraph(
        '解决方案：将指标按语义分为3组——预测精度（R²_mean, RMSE_mean, MAPE_mean）、'
        '稳定性（R²_std, RMSE_std）、泛化能力（tail_q10, tail_q90）。'
        '组间权重基于领域知识主观设定（精度50%、稳定性25%、泛化25%），'
        '组内以熵权法客观分配。MAE_mean因与RMSE_mean高度共线（r=0.997）被排除。'
        '这一"组间主观+组内客观"的混合策略在方法论上同时防御了"等权无信息"和"维度通胀"两类审稿批评。'
    )

    weights_grp_df = pd.read_csv(TABLES / 'grouped_weights.csv')
    add_styled_table(doc,
        ['指标', '方向', '所属组', '组权重', '组内熵值 e', '组内权重', '最终权重 w'],
        [[w['指标'], w['方向'], w['所属组'],
          f"{w['组权重']:.2f}", f"{w['组内熵值 e']:.4f}", f"{w['组内权重']:.4f}",
          f"{w['最终权重 w']:.4f}"]
         for _, w in weights_grp_df.iterrows()]
    )
    doc.add_paragraph(
        '权重结构：预测精度组内R²/RMSE/MAPE几乎等权（~33.3%），各获得约16.7%总权重；'
        '稳定性组和泛化组内各约50/50分配，各指标约12.5%。'
        'R²（16.35%）获得与RMSE（16.67%）和MAPE（16.99%）同等的权重——'
        '在领域知识层面，这是合理的：拟合优度不应被误差指标淹没。'
    )

    doc.add_heading('5.2 标准熵权法（参考）', level=2)
    doc.add_paragraph(
        '标准熵权法对8个指标独立赋权，权重分布极均匀（0.112–0.133），'
        '极差仅0.020，基本丧失区分能力。其价值在于：作为"无信息先验"验证层次熵权法的排名不受权重微调影响。'
    )
    weights_ent_df = pd.read_csv(TABLES / 'entropy_weights.csv')
    add_styled_table(doc,
        ['指标', '方向', '熵值 e', '权重 w'],
        [[w['指标'], w['方向'], f"{w['熵值 e']:.4f}", f"{w['权重 w']:.4f}"]
         for _, w in weights_ent_df.iterrows()]
    )

    doc.add_heading('5.3 CRITIC法（敏感性分析）', level=2)
    doc.add_paragraph(
        'CRITIC通过C_j = σ_j × Σ(1-|r_jk|)同时考虑对比强度和冲突度。但本研究n=9模型，'
        '相关系数估计极其不稳定（95%CI宽度0.2–0.8），CRITIC的冲突度A_j对采样噪声高度敏感。'
        '此外，所有8个指标|r|>0.93（模型质量本质上是1维的），CRITIC的前提——存在独立信息维度——不成立。'
        '因此CRITIC仅作为敏感性分析参考，不作为主方案。'
    )
    weights_crit_df = pd.read_csv(TABLES / 'critic_weights.csv')
    add_styled_table(doc,
        ['指标', '方向', '对比强度 σ', '冲突度 A', '信息量 C', '权重 w'],
        [[w['指标'], w['方向'],
          f"{w['对比强度 σ']:.4f}", f"{w['冲突度 A']:.4f}",
          f"{w['信息量 C']:.4f}", f"{w['权重 w']:.4f}"]
         for _, w in weights_crit_df.iterrows()]
    )

    doc.add_heading('5.4 三方案排名对比', level=2)
    grouped_df = pd.read_csv(TABLES / 'topsis_grouped.csv')
    entropy_df = pd.read_csv(TABLES / 'topsis_entropy.csv')
    critic_df = pd.read_csv(TABLES / 'topsis_critic.csv')

    merged = grouped_df[['模型', '排名', '贴近度 C_i']].rename(
        columns={'排名': '层次熵权排名', '贴近度 C_i': 'C_i_grp'}
    )
    merged = merged.merge(
        entropy_df[['模型', '排名', '贴近度 C_i']].rename(
            columns={'排名': '标准熵权排名', '贴近度 C_i': 'C_i_ent'}
        ), on='模型'
    )
    merged = merged.merge(
        critic_df[['模型', '排名', '贴近度 C_i']].rename(
            columns={'排名': 'CRITIC排名', '贴近度 C_i': 'C_i_crit'}
        ), on='模型'
    )
    merged['排名极差'] = merged[['层次熵权排名', '标准熵权排名', 'CRITIC排名']].max(axis=1) - \
                        merged[['层次熵权排名', '标准熵权排名', 'CRITIC排名']].min(axis=1)
    merged = merged.sort_values('层次熵权排名').reset_index(drop=True)

    add_styled_table(doc,
        ['主排名', '模型', 'C_i (层次熵权)', '标准熵权', 'C_i (熵权)', 'CRITIC', 'C_i (CRITIC)', '极差'],
        [[str(r['层次熵权排名']), r['模型'],
          f"{r['C_i_grp']:.4f}",
          str(r['标准熵权排名']), f"{r['C_i_ent']:.4f}",
          str(r['CRITIC排名']), f"{r['C_i_crit']:.4f}",
          str(r['排名极差'])]
         for _, r in merged.iterrows()]
    )

    doc.add_paragraph(
        '三方案排名关键发现：\n'
        '• 层次熵权法与标准熵权法排名完全一致（Spearman ρ=1.0），'
        '说明组间赋权仅改变了权重的"名义分配"，对最终排名无实质影响——'
        '这恰恰证明了排名的稳健性：无论你将R²称为12.8%还是16.4%，它都不改变谁是第一名\n'
        '• TabPFN在两种熵权方案下均为第1（C_i≈0.94），零超参基础模型在8项指标上全面领跑\n'
        '• 中游模型（RF/LightGBM/SVR/GPR）在三方案下排名完全一致（极差=0），极其稳健\n'
        '• CRITIC因小样本敏感性将TabPFN排至第3，此差异应在论文中作为方法论局限性讨论\n'
        '• 三种方法均确认：非线性模型（C_i>0.67）远超线性基线（C_i<0.03），差距约30倍'
    )

    # ============================================================
    # 6. SHAP 可解释性
    # ============================================================
    doc.add_heading('6. SHAP 可解释性分析', level=1)

    doc.add_heading('6.1 多重共线性诊断（VIF）', level=2)
    doc.add_paragraph('共11个特征VIF > 10（高度共线），主要为孔结构参数耦合（SBET/Vtotal/Vmicro/Vmeso）和碳化参数。')
    add_styled_table(doc,
        ['特征', 'VIF', '说明'],
        [
            ['Vtotal', '∞', '与其他孔结构参数完全共线'],
            ['Vmicro', '∞', '与其他孔结构参数完全共线'],
            ['Vmeso', '∞', 'Vtotal − Vmicro，确定性关系'],
            ['act2 duration', '∞', '数据极稀疏（仅9条非零）'],
            ['act2 temp', '∞', '同上'],
            ['carb2 duration', '∞', '数据分布集中'],
            ['carb2 temp', '∞', '同上'],
            ['SBET', '153.62', 'R²=0.9935，与孔容高度耦合'],
            ['1/T', '41.19', '与Temperature确定性转换'],
            ['Temperature', '35.32', '与1/T共线'],
            ['carb1 temp', '13.48', '中度共线'],
        ]
    )
    doc.add_paragraph(
        '归因策略：对高度共线特征簇（如SBET/Vtotal/Vmicro）除报告个体SHAP外，额外报告簇级SHAP，避免武断声称单一孔结构参数更重要。'
    )

    doc.add_heading('6.2 Spearman特征聚类', level=2)
    doc.add_paragraph('Spearman相关系数矩阵经Ward层次聚类，共分为9个特征簇：')
    doc.add_paragraph(
        '• 簇1（6特征）：SBET, Vtotal, Vmicro, Temperature, Vmeso, 1/T — 孔结构与温度组\n'
        '• 簇2（4特征）：Pressure, carb1 temp, carb1 duration, T·ln(P) — 碳化条件与吸附条件组\n'
        '• 簇3（2特征）：carb2 temp, carb2 duration — 二次碳化参数\n'
        '• 簇4（2特征）：MgO Mass Ratio, MgO Surface Density — MgO负载量组\n'
        '• 簇5-9（各1特征）：act1 temp, act1 duration, Microporosity, act2 temp, act2 duration — 独立特征'
    )

    doc.add_heading('6.3 GBDT SHAP特征重要性', level=2)
    shap_df = pd.read_csv(TABLES / 'shap_importance_gbdt.csv')
    top10_shap = shap_df.head(10)
    add_styled_table(doc,
        ['排名', '特征', 'SHAP重要性 (mean|SHAP|)'],
        [[str(i+1), row['feature'], f"{row['shap_importance_mean']:.2f}"]
         for i, row in top10_shap.iterrows()]
    )
    doc.add_paragraph(
        'Top 1 Pressure（SHAP=19.17）和Top 2 Microporosity（SHAP=13.80）远高于后续特征，'
        '说明吸附操作条件和微孔结构是CO₂吸附量的两大核心驱动力。'
    )

    doc.add_heading('6.4 TabPFN排列特征重要性', level=2)
    perm_df = pd.read_csv(TABLES / 'permutation_importance_tabpfn.csv')
    top10_perm = perm_df.head(10)
    add_styled_table(doc,
        ['排名', '特征', '重要性均值', '标准差'],
        [[str(i+1), row['feature'], f"{row['importance_mean']:.2f}", f"{row['importance_std']:.2f}"]
         for i, row in top10_perm.iterrows()]
    )

    doc.add_heading('6.5 归因一致性', level=2)
    doc.add_paragraph(
        'TabPFN排列重要性 vs GBDT SHAP重要性 Spearman ρ = 0.8796（p ≈ 0），'
        '满足ρ > 0.6的阈值。两种截然不同的模型范式在特征归因上高度一致，'
        '表明所识别的关键特征（Pressure, Microporosity, Temperature）是CO₂吸附的鲁棒驱动因素，'
        '不依赖于特定模型架构。'
    )

    # ============================================================
    # 7. 论文图表
    # ============================================================
    doc.add_heading('7. 论文图表', level=1)

    doc.add_heading('7.1 已完成', level=2)
    add_styled_table(doc,
        ['图号', '内容', '状态', '文件'],
        [
            ['Figure 1', 'Spearman相关热力图（17特征，Ward聚类，RdBu配色，18×17", 300 DPI）', '已完成',
             'outputs/figures/Figure_1_Clustermap.png (1.1 MB)'],
        ]
    )
    doc.add_paragraph('Figure 1 描述：特征间Spearman秩相关矩阵，以Ward层次聚类重排特征顺序，RdBu柔和配色，标注VIF>10的共线特征簇。')

    doc.add_heading('7.2 待完成', level=2)
    add_styled_table(doc,
        ['图号', '计划内容'],
        [
            ['Figure 2', '不同前驱体类型的CO₂吸附量分布（箱线图/小提琴图）'],
            ['Figure 3', '模型性能对比柱状图（R² + RMSE，9模型，标注线性-非线性-基础模型性能差距 + TOPSIS贴近度）'],
            ['Figure 4', '预测值 vs 实验值散点图，按SBET着色'],
            ['Figure 5', 'SHAP summary beeswarm图（标注共线特征簇）'],
            ['Figure 6', 'Top 3特征的SHAP依赖图'],
            ['Figure 7', '残差分析 2×2（残差vs拟合值、Q-Q图、直方图、Cook距离）'],
            ['Figure 8', 'Top 4特征的部分依赖图 + ICE'],
        ]
    )

    # ============================================================
    # 8. 文件清单
    # ============================================================
    doc.add_heading('8. 输出文件清单', level=1)

    doc.add_heading('8.1 表格（outputs/tables/）', level=2)
    doc.add_paragraph(
        '• cv_results.json — 9个模型的完整8指标CV结果\n'
        '• topsis_grouped.csv — 层次熵权法TOPSIS排名表 ★ 论文主方案\n'
        '• grouped_weights.csv — 层次熵权法权重分解（组权重+组内熵权）\n'
        '• topsis_entropy.csv — 标准熵权法TOPSIS排名表（参考）\n'
        '• entropy_weights.csv — 标准熵权法各指标权重\n'
        '• topsis_critic.csv — CRITIC法TOPSIS排名表（敏感性分析）\n'
        '• critic_weights.csv — CRITIC法各指标权重\n'
        '• shap_importance_gbdt.csv — GBDT SHAP特征重要性（27特征）\n'
        '• shap_values_gbdt.csv — 341条样本×27特征完整SHAP值矩阵\n'
        '• permutation_importance_tabpfn.csv — TabPFN排列重要性（27特征）\n'
        '• vif_table.csv — VIF共线性诊断表\n'
        '• feature_clusters.csv — Spearman聚类结果（9簇）\n'
        '• spearman_correlation.csv — 19×19 Spearman相关矩阵\n'
        '• consistency_comparison.csv — TabPFN vs GBDT排名一致性比较'
    )

    doc.add_heading('8.2 模型（outputs/models/）', level=2)
    doc.add_paragraph(
        '9个最终模型已保存为.pkl文件（Total ~252 MB）：\n'
        'TabPFN (223.6 MB), RF (10.1 MB), LightGBM (2.2 MB), GPR (1.0 MB), '
        'XGBoost (660 KB), GBDT (605 KB), SVR (65 KB), Ridge (8 KB), Lasso (8 KB)'
    )

    doc.add_heading('8.3 源代码（src/）', level=2)
    doc.add_paragraph(
        '共12个.py文件，流水线为：data_loader → preprocessing → feature_engineering → '
        'models → train → evaluate → topsis → shap_analysis → plotting'
    )

    # ============================================================
    # 9. 验证标准
    # ============================================================
    doc.add_heading('9. 验证标准达成情况', level=1)
    add_styled_table(doc,
        ['验证标准', '阈值', '实际值', '状态'],
        [
            ['XGBoost/LightGBM 嵌套CV R²', '> 0.80', '0.9449 / 0.9350', '✓ 通过'],
            ['非线性-线性 R² 差距', '≥ 0.05', '0.592', '✓ 通过'],
            ['TabPFN(零超参) ≥ RF(调优后)', '≥ RF', '0.9692 vs 0.9503', '✓ 通过'],
            ['预处理数据泄露', '零容忍', 'Pipeline内部化，已根除', '✓ 通过'],
            ['TabPFN排列重要性 vs GBDT SHAP ρ', '> 0.6', '0.8796', '✓ 通过'],
            ['TOPSIS三方案排名稳健性', '极差≤2', 'TabPFN极差2, 其余≤1', '✓ 通过'],
            ['Vmicro填补 vs 完整样本 R²差距', '< 0.10', '待验证', '□ 未完成'],
        ]
    )

    # ============================================================
    # 10. 下一步
    # ============================================================
    doc.add_heading('10. 下一步计划', level=1)
    doc.add_paragraph(
        '1. 完成 Figure 2–8 论文图表（plotting.py）\n'
        '2. Vmicro填补稳健性检验（Vmicro填补 vs 仅完整样本 R²差距 < 0.10）\n'
        '3. 完成 7 个 Jupyter Notebook（01_EDA → 07_SHAP）\n'
        '4. 撰写论文初稿\n'
        '5. 创建 requirements.txt'
    )

    # 保存
    report_path = ROOT / "outputs" / "项目综合进展报告.docx"
    doc.save(str(report_path))
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
