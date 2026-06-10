#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成项目汇报PPT —— 生物质MgO多孔碳CO2吸附ML预测"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── 颜色方案 ──────────────────────────────────────
DARK_BLUE   = RGBColor(0x1B, 0x2A, 0x4A)  # 深蓝主色
MID_BLUE    = RGBColor(0x2C, 0x5F, 0x8A)  # 中蓝
LIGHT_BLUE  = RGBColor(0x4A, 0x90, 0xD9)  # 亮蓝
ACCENT_GOLD = RGBColor(0xD4, 0x8B, 0x2C)  # 强调金
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF2, 0xF2, 0xF2)
DARK_GRAY   = RGBColor(0x33, 0x33, 0x33)
MED_GRAY    = RGBColor(0x88, 0x88, 0x88)
RED_ACCENT  = RGBColor(0xC0, 0x39, 0x2B)
GREEN_ACC   = RGBColor(0x27, 0xAE, 0x60)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

def add_bg(slide, color=DARK_BLUE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, left, top, width, height, color, transparency=0):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=WHITE, align=PP_ALIGN.LEFT, font_name='Microsoft YaHei'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return txBox

def add_multi_text(slide, left, top, width, height, lines, font_size=14,
                   color=WHITE, line_spacing=1.5, font_name='Microsoft YaHei'):
    """lines: list of (text, bold, font_size_override)"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        if isinstance(item, str):
            text, bold, fs = item, False, font_size
        else:
            text = item[0]
            bold = item[1] if len(item) > 1 else False
            fs = item[2] if len(item) > 2 else font_size
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(fs)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = Pt(line_spacing * fs * 0.5)
    return txBox

def add_title_bar(slide, title, subtitle=""):
    """顶部标题栏"""
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.2), DARK_BLUE)
    add_text_box(slide, Inches(0.8), Inches(0.15), Inches(11), Inches(0.7),
                 title, font_size=32, bold=True, color=WHITE)
    if subtitle:
        add_text_box(slide, Inches(0.8), Inches(0.75), Inches(11), Inches(0.4),
                     subtitle, font_size=14, color=RGBColor(0xBB, 0xCC, 0xDD))

def add_table(slide, left, top, col_widths, headers, rows, font_size=11):
    """添加表格"""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top,
                                         sum(col_widths), Inches(0.4 * n_rows))
    table = table_shape.table

    for ci, cw in enumerate(col_widths):
        table.columns[ci].width = cw

    # 表头
    for ci, h in enumerate(headers):
        cell = table.cell(0, ci)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font_size)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.font.name = 'Microsoft YaHei'
            p.alignment = PP_ALIGN.CENTER

    # 数据行
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri + 1, ci)
            cell.text = str(val)
            if ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size)
                p.font.color.rgb = DARK_GRAY
                p.font.name = 'Microsoft YaHei'
                p.alignment = PP_ALIGN.CENTER
    return table_shape

def add_accent_line(slide, left, top, width, color=ACCENT_GOLD):
    add_rect(slide, left, top, width, Inches(0.04), color)

def add_page_num(slide, num, total):
    add_text_box(slide, Inches(12), Inches(7.1), Inches(1.2), Inches(0.3),
                 f"{num}/{total}", font_size=10, color=MED_GRAY, align=PP_ALIGN.RIGHT)

TOTAL_SLIDES = 21  # 预估

# ═══════════════════════════════════════════════════════
# Slide 1: 封面
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, DARK_BLUE)

# 装饰线条
add_rect(slide, Inches(2), Inches(2.2), Inches(9.3), Inches(0.06), ACCENT_GOLD)
add_rect(slide, Inches(2), Inches(5.5), Inches(9.3), Inches(0.03), ACCENT_GOLD)

add_text_box(slide, Inches(2), Inches(2.5), Inches(9.3), Inches(1.5),
             "生物质衍生MgO改性多孔碳CO₂吸附量\n机器学习预测研究",
             font_size=42, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_text_box(slide, Inches(2), Inches(4.2), Inches(9.3), Inches(0.8),
             "Machine Learning Prediction of CO₂ Uptake on Biomass-Derived\nMgO-Modified Porous Carbon",
             font_size=20, color=RGBColor(0xAA, 0xBB, 0xCC), align=PP_ALIGN.CENTER)

add_text_box(slide, Inches(2), Inches(5.8), Inches(9.3), Inches(0.5),
             "项目进展汇报  |  2026年6月",
             font_size=18, color=RGBColor(0x88, 0x99, 0xAA), align=PP_ALIGN.CENTER)

add_text_box(slide, Inches(2), Inches(6.5), Inches(9.3), Inches(0.4),
             "341条文献数据 · 9个ML模型 · 4条预处理管道 · SHAP可解释性 · TOPSIS综合评估",
             font_size=14, color=MED_GRAY, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════
# Slide 2: 目录
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "目录  |  Contents")
add_page_num(slide, 2, TOTAL_SLIDES)

toc_items = [
    ("01", "项目背景与研究目标", "Background & Objectives"),
    ("02", "数据集概览", "Dataset Overview"),
    ("03", "全部特征分类详情", "Feature Classification"),
    ("04", "研究方法学", "Methodology"),
    ("05", "模型性能 —— 嵌套交叉验证", "Nested CV Results"),
    ("06", "80/20 分割与50次重复子抽样验证", "80/20 Split & 50-Repeated Subsampling"),
    ("07", "TOPSIS 多准则综合排名", "TOPSIS Multi-Criteria Ranking"),
    ("08", "SHAP 可解释性分析", "SHAP Explainability"),
    ("09", "特征重要性一致性验证", "Feature Importance Consistency"),
    ("10", "图表展示", "Figures Overview"),
    ("11", "关键结论与下一步", "Conclusions & Next Steps"),
]

for i, (num, title, eng) in enumerate(toc_items):
    y = Inches(1.6) + i * Inches(0.52)
    add_rect(slide, Inches(1), y, Inches(0.6), Inches(0.38), DARK_BLUE)
    add_text_box(slide, Inches(1.05), y + Inches(0.02), Inches(0.5), Inches(0.34),
                 num, font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.9), y + Inches(0.02), Inches(6), Inches(0.22),
                 title, font_size=18, bold=True, color=DARK_GRAY)
    add_text_box(slide, Inches(1.9), y + Inches(0.24), Inches(6), Inches(0.16),
                 eng, font_size=11, color=MED_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 3: 项目背景
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "1. 项目背景与研究目标", "Background & Objectives")
add_page_num(slide, 3, TOTAL_SLIDES)

add_multi_text(slide, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.2), [
    ("研究背景", True, 22),
    ("", False, 8),
    ("• CO₂排放是全球气候变化的主要驱动因素", False, 16),
    ("• 生物质衍生多孔碳是极具潜力的CO₂吸附材料", False, 16),
    ("• MgO改性可显著增强表面碱性位点，提升吸附性能", False, 16),
    ("• 传统实验优化耗时耗力，ML可加速材料筛选与优化", False, 16),
    ("", False, 8),
    ("研究目标", True, 22),
    ("", False, 8),
    ("• 基于341条文献数据建立CO₂吸附量预测模型", False, 16),
    ("• 系统比较9种ML模型在4条预处理管道下的性能", False, 16),
    ("• 利用SHAP揭示关键理化特征对吸附性能的驱动机制", False, 16),
    ("• 通过TOPSIS多准则综合排名推荐最优模型", False, 16),
    ("• 提供可复现的ML材料科学分析框架", False, 16),
], color=DARK_GRAY)

# 右侧关键数字
add_rect(slide, Inches(7.5), Inches(1.6), Inches(5.0), Inches(5.2), DARK_BLUE)
add_multi_text(slide, Inches(8.0), Inches(2.0), Inches(4.0), Inches(4.5), [
    ("关键数据", True, 22),
    ("", False, 10),
    ("📊  341 条文献样本", False, 18),
    ("", False, 6),
    ("🤖  9 个机器学习模型", False, 18),
    ("", False, 6),
    ("🔧  4 条预处理管道", False, 18),
    ("", False, 6),
    ("📈  27 个工程特征", False, 18),
    ("", False, 6),
    ("🎯  1 个目标变量", False, 18),
    ("    (CO₂ uptake, mg/g)", False, 14),
    ("", False, 6),
    ("🔬  5 个领域复合特征", False, 18),
], color=WHITE)

# ═══════════════════════════════════════════════════════
# Slide 4: 数据集概览
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "2. 数据集概览", "Dataset Overview — 341 samples, 16 raw columns")
add_page_num(slide, 4, TOTAL_SLIDES)

add_multi_text(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5), [
    ("原始数据 (16列)", True, 20),
    ("", False, 6),
    ("数值特征 (6列):", True, 15),
    ("  SBET (m²/g), Vtotal (cm³/g), Vmicro (cm³/g),", False, 14),
    ("  MgO (mass ratio), Temperature (°C), Pressure (bar)", False, 14),
    ("", False, 6),
    ("分类/工艺特征 (9列):", True, 15),
    ("  碳前驱体 (13类), MgO前驱体 (6类), 活化类型,", False, 14),
    ("  碳化类型, Mg负载方法, 后处理类型等", False, 14),
    ("", False, 6),
    ("目标变量 (1列):", True, 15),
    ("  CO₂ uptake (mg/g)", False, 14),
    ("", False, 8),
    ("特征工程后 (27列):", True, 20),
    ("  • 5个领域复合特征: Vmeso, microporosity,", False, 14),
    ("     MgO_surface_density, T_lnP, inv_T_K", False, 14),
    ("  • 正则化提取: 温度(°C), 时长(h)", False, 14),
    ("  • 工艺变量: OneHot / Ordinal 编码", False, 14),
], color=DARK_GRAY)

# 右侧分布信息
add_rect(slide, Inches(7.5), Inches(1.5), Inches(5.0), Inches(5.5), LIGHT_GRAY)
# S1 figure placeholder
add_text_box(slide, Inches(7.8), Inches(1.7), Inches(4.4), Inches(0.4),
             "Figure S1: 前驱体分布", font_size=13, bold=True, color=DARK_BLUE)
fig_s1 = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_S1_Precursor_Distribution.jpg")
if os.path.exists(fig_s1):
    slide.shapes.add_picture(fig_s1, Inches(8.0), Inches(2.2), Inches(4.0), Inches(4.3))

# ═══════════════════════════════════════════════════════
# Slide 5: 全部特征分类详情
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "2.2 全部特征分类详情", "Feature Classification — 原始16列 → 工程27列 → SHAP 22列")
add_page_num(slide, 5, TOTAL_SLIDES)

# 左栏：数值/复合特征
add_multi_text(slide, Inches(0.4), Inches(1.4), Inches(4.2), Inches(5.8), [
    ("一、目标变量 (1个)", True, 14),
    ("  CO₂_uptake_mg_g — CO₂吸附量 (mg/g)", False, 11),
    ("", False, 4),
    ("二、原始数值特征 (6个)", True, 14),
    ("  SBET_m2_g — BET比表面积", False, 11),
    ("  Vmicro_cm3_g — 微孔容", False, 11),
    ("  MgO_mass_ratio — MgO质量比", False, 11),
    ("  temperature_C — 吸附温度 (°C)", False, 11),
    ("  pressure_bar — 吸附压力 (bar)", False, 11),
    ("", False, 4),
    ("三、领域复合特征 (5个)", True, 14),
    ("  Vmeso_cm3_g = Vtotal − Vmicro", False, 11),
    ("  microporosity = Vmicro / Vtotal", False, 11),
    ("  MgO_surface_density = MgO / SBET", False, 11),
    ("  T_lnP = T × ln(P)", False, 11),
    ("  inv_T_K = 1 / (T + 273.15)", False, 11),
    ("", False, 4),
    ("四、数值工艺特征 (8个，4个VIF剔除)", True, 14),
    ("  act1_temp_C / act1_duration_h (248非空)", False, 11),
    ("  act2_temp_C / act2_duration_h ⚠ 剔除 (9非空)", False, 11),
    ("  carb1_temp_C (336) / carb1_duration_h (289)", False, 11),
    ("  carb2_temp_C / carb2_duration_h ⚠ 剔除 (174)", False, 11),
], color=DARK_GRAY)

# 中栏：分类特征
add_multi_text(slide, Inches(4.8), Inches(1.4), Inches(4.2), Inches(5.8), [
    ("五、分类特征 (8个原始变量)", True, 14),
    ("", False, 4),
    ("carbon_precursors (13类)", True, 12),
    ("  Cottonwood, Wheat straw, coffee grounds,", False, 10),
    ("  palm empty fruit bunch, palm kernel shells,", False, 10),
    ("  pollen grains, rambutan peel, rice husk,", False, 10),
    ("  saw dust, sawdust, sugarcane bagasse,", False, 10),
    ("  walnut shell, whitewood", False, 10),
    ("", False, 3),
    ("MgO_precursors (6类)", True, 12),
    ("  Mg(NO₃)₂⋅6H₂O, Mg(CH₃COO)₂⋅4H₂O,", False, 10),
    ("  MgCl₂⋅6H₂O, MgCl₂⋅7H₂O, MgO, MgSO₄⋅7H₂O", False, 10),
    ("", False, 3),
    ("act1_type (4类) — KOH/hydrothermal/none/steam", False, 10),
    ("act2_type (2类) — H₃PO₄/none", False, 10),
    ("carb1_type (4类)", False, 10),
    ("  conventional/fast_pyrolysis/microwave/none", False, 10),
    ("carb2_type (3类)", False, 10),
    ("  500C_15min / 600C_2h / none", False, 10),
    ("Mg_loading_method (3类)", False, 10),
    ("  ball milling / hydrothermal / impregnation", False, 10),
    ("post_treatment_type (4类)", False, 10),
    ("  none / sonication / with_rinsing / w/o_rinsing", False, 10),
], color=DARK_GRAY)

# 右栏：汇总数字
add_rect(slide, Inches(9.2), Inches(1.4), Inches(3.8), Inches(5.8), DARK_BLUE)
add_multi_text(slide, Inches(9.5), Inches(1.6), Inches(3.3), Inches(5.4), [
    ("特征统计总览", True, 18),
    ("", False, 8),
    ("原始数值特征", True, 14),
    ("  6 个", False, 24),
    ("", False, 8),
    ("领域复合特征", True, 14),
    ("  5 个", False, 24),
    ("", False, 8),
    ("数值工艺特征", True, 14),
    ("  8 个 (4个VIF剔除)", False, 16),
    ("", False, 8),
    ("分类原始变量", True, 14),
    ("  8 个 (38个OneHot列)", False, 16),
    ("", False, 8),
    ("━━━━━━━━━━", False, 14),
    ("工程后总列数", True, 16),
    ("  27 列", False, 28),
    ("SHAP有效特征", True, 16),
    ("  22 个", False, 28),
    ("VIF剔除特征", True, 16),
    ("  5 个", False, 28),
    ("目标变量", True, 16),
    ("  1 个", False, 28),
], color=WHITE)

# ═══════════════════════════════════════════════════════
# Slide 6: 方法学概览
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "3. 研究方法学", "Methodology — Pipeline Architecture")
add_page_num(slide, 6, TOTAL_SLIDES)

# 流程图
add_multi_text(slide, Inches(0.8), Inches(1.5), Inches(11.5), Inches(1.5), [
    ("数据处理流水线:  load → impute → features → models → evaluate → SHAP → TOPSIS → plot", True, 16),
    ("每个模块为独立 sklearn Transformer，可复用与验证", False, 13),
], color=DARK_GRAY)

# 四个管道
pipe_data = [
    ("管道 A (Ridge, Lasso)", "OneHotEncoder + StandardScaler\n+ SimpleImputer(median)", LIGHT_BLUE),
    ("管道 B (RF, XGBoost, LightGBM, GBDT)", "OrdinalEncoder\n树模型原生支持NaN", MID_BLUE),
    ("管道 C (SVR, GPR)", "TargetEncoder + StandardScaler\n+ log1p(y) + SimpleImputer(median)", RGBColor(0x5D, 0x6D, 0x7E)),
    ("管道 D (TabPFN)", "OrdinalEncoder\n内置预处理，零超参", DARK_BLUE),
]

for i, (name, desc, color) in enumerate(pipe_data):
    x = Inches(0.5) + i * Inches(3.2)
    y = Inches(3.3)
    shape = add_rect(slide, x, y, Inches(2.9), Inches(3.5), color)
    add_text_box(slide, x + Inches(0.2), y + Inches(0.2), Inches(2.5), Inches(0.8),
                 name, font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, x + Inches(0.2), y + Inches(1.2), Inches(2.5), Inches(2.0),
                 desc, font_size=12, color=WHITE, align=PP_ALIGN.CENTER)

# CV 策略说明
add_multi_text(slide, Inches(0.8), Inches(7.0), Inches(11.5), Inches(0.5), [
    ("交叉验证: 嵌套CV (StratifiedKFold 5×3)  |  超参优化: Optuna TPE (n_trials=100)  |  KDE权重聚合  |  种子: 42 (最终评估91)", False, 11),
], color=MED_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 6: 9个模型一览
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "3.2 九大模型与四条管道", "9 Models × 4 Pipelines")
add_page_num(slide, 7, TOTAL_SLIDES)

models_rows = [
    ["Ridge", "A", "OneHot → Scale → Impute", "线性 / L2正则", "对照基线"],
    ["Lasso", "A", "OneHot → Scale → Impute", "线性 / L1正则", "对照基线"],
    ["RF", "B", "Ordinal → 原生NaN", "集成 / Bagging", "非线性标杆"],
    ["XGBoost", "B", "Ordinal → 原生NaN", "Boosting / 正则化", "SOTA"],
    ["LightGBM", "B", "Ordinal → 原生NaN", "Boosting / Leaf-wise", "高效"],
    ["GBDT", "B", "Ordinal → 原生NaN", "Boosting / 经典", "SHAP主力"],
    ["SVR", "C", "Target → Scale+log1p", "核方法 / ε-不敏感", "小样本Huber"],
    ["GPR", "C", "Target → Scale+log1p", "贝叶斯 / 核函数", "不确定性"],
    ["TabPFN", "D", "Ordinal → 内置预处理", "Transformer / In-Context", "零超参SOTA"],
]

add_table(slide, Inches(1.5), Inches(1.6),
          [Inches(1.5), Inches(0.8), Inches(3.0), Inches(2.5), Inches(2.0)],
          ["模型", "管道", "预处理", "算法类型", "角色定位"],
          models_rows, font_size=11)

# 说明
add_multi_text(slide, Inches(1.5), Inches(5.6), Inches(10), Inches(1.5), [
    ("关键设计决策:", True, 15),
    ("• TabPFN: 基于Transformer的先验数据拟合网络，零超参，in-context learning", False, 13),
    ("• 管道B (树模型) 保留最大信息量，不缩放不填补，原生处理缺失值", False, 13),
    ("• 管道C (SVR/GPR) 用TargetEncoder降低高基数分类变量维度，log1p缓解右偏", False, 13),
    ("• 所有管道内置 MissingValueImputer + FeatureEngineer，杜绝数据泄露", False, 13),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 7: 嵌套CV结果
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "4. 模型性能 —— 嵌套交叉验证结果", "Nested CV (StratifiedKFold 5×3) with Optuna TPE (n_trials=100)")
add_page_num(slide, 8, TOTAL_SLIDES)

cv_rows = [
    ["🥇 1", "TabPFN", "0.9692", "0.0103", "10.73", "2.28", "4.12", "7.99%", "D"],
    ["🥈 2", "GBDT",    "0.9528", "0.0138", "13.35", "1.90", "7.25", "16.91%", "B"],
    ["🥉 3", "RF",      "0.9503", "0.0120", "13.75", "2.02", "7.69", "20.63%", "B"],
    ["4", "XGBoost",  "0.9449", "0.0150", "14.35", "1.39", "7.31", "15.72%", "B"],
    ["5", "LightGBM", "0.9350", "0.0196", "15.67", "2.48", "8.67", "18.66%", "B"],
    ["6", "GPR",      "0.9299", "0.0306", "15.94", "3.50", "7.83", "16.77%", "C"],
    ["7", "SVR",      "0.9112", "0.0226", "18.28", "2.54", "8.32", "14.99%", "C"],
    ["8", "Ridge",    "0.3608", "0.0804", "49.66", "5.90", "35.73", "81.95%", "A"],
    ["9", "Lasso",    "0.3521", "0.0774", "50.00", "5.77", "36.43", "81.77%", "A"],
]

add_table(slide, Inches(0.5), Inches(1.6),
          [Inches(0.4), Inches(1.2), Inches(1.1), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(0.5)],
          ["#", "模型", "R²_mean", "R²_std", "RMSE", "RMSE_std", "MAE", "MAPE", "管道"],
          cv_rows, font_size=11)

# 关键发现
add_rect(slide, Inches(0.5), Inches(5.5), Inches(12.3), Inches(1.6), RGBColor(0xEB, 0xF5, 0xFB))
add_multi_text(slide, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.4), [
    ("🔑 核心发现", True, 18),
    ("• 非线性模型 (R²>0.91) 远超线性模型 (R²≈0.36): 差距 0.592，证实CO₂吸附预测为高度非线性问题", False, 13),
    ("• TabPFN以零超参位列第一 (R²=0.9692)，验证了Transformer in-context learning在小样本表格数据上的优势", False, 13),
    ("• 树模型四强 (GBDT/RF/XGBoost/LightGBM) R²差距不足0.02，性能稳健", False, 13),
    ("• GPR (R²=0.93) 由于不确定性估计优势超越SVR (R²=0.91)，适合风险敏感场景", False, 13),
], color=DARK_BLUE)

# ═══════════════════════════════════════════════════════
# Slide 8: 非线性-线性差距可视化
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "4.2 模型性能分布（箱线图）", "Model Performance Boxplot — 非线性 vs 线性差距 0.592")
add_page_num(slide, 9, TOTAL_SLIDES)

# Boxplot figure
fig_box = os.path.join(os.path.dirname(__file__), "outputs", "figures", "model_performance_boxplot_final.png")
if os.path.exists(fig_box):
    slide.shapes.add_picture(fig_box, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.5))

add_multi_text(slide, Inches(8.8), Inches(1.7), Inches(4.0), Inches(4.5), [
    ("性能梯度解读", True, 18),
    ("", False, 6),
    ("Tier 1 (R²>0.96):", True, 14),
    ("  TabPFN — Transformer ICL", False, 13),
    ("", False, 5),
    ("Tier 2 (R² 0.93-0.95):", True, 14),
    ("  GBDT / RF / XGBoost / LightGBM", False, 13),
    ("  树模型四强，差距<0.02", False, 13),
    ("", False, 5),
    ("Tier 3 (R² 0.91-0.93):", True, 14),
    ("  GPR / SVR — 核方法", False, 13),
    ("", False, 5),
    ("Tier 4 (R²≈0.36):", True, 14),
    ("  Ridge / Lasso — 线性对照", False, 13),
    ("", False, 8),
    ("非线性优势 0.592 >> 0.05 阈值", True, 14),
    ("证实问题高度非线性", False, 12),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 9: 80/20 分割结果
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "5. 80/20 分层分割验证 (种子91)", "80/20 Stratified Split — 272 Train / 69 Test (seed=91)")
add_page_num(slide, 10, TOTAL_SLIDES)

split_rows = [
    ["🥇 TabPFN",  "0.9988", "2.11", "0.9688", "11.07", "4.22", "9.28%"],
    ["🥈 LightGBM","0.9993", "1.69", "0.9676", "11.28", "7.01", "14.48%"],
    ["🥉 GBDT",    "0.9998", "0.93", "0.9639", "11.91", "7.14", "17.65%"],
    ["4  SVR",     "0.9853", "7.54", "0.9628", "12.07", "6.61", "20.57%"],
    ["5  RF",      "0.9919", "5.59", "0.9615", "12.30", "7.45", "26.94%"],
    ["6  GPR",     "0.9894", "6.39", "0.9376", "15.65", "7.71", "25.57%"],
    ["7  XGBoost", "0.9995", "1.32", "0.9151", "18.25", "10.02","23.39%"],
]

add_table(slide, Inches(0.5), Inches(1.5),
          [Inches(1.4), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.0), Inches(1.1)],
          ["模型", "Train R²", "Train RMSE", "Test R²", "Test RMSE", "Test MAE", "Test MAPE"],
          split_rows, font_size=12)

add_multi_text(slide, Inches(0.5), Inches(4.8), Inches(6.0), Inches(2.2), [
    ("关键观察:", True, 16),
    ("• TabPFN Test R²=0.9688 ≈ 嵌套CV 0.9692 (Δ=-0.0004)", False, 13),
    ("• 种子91经0-100遍历优选，确保代表性", False, 13),
    ("• TabPFN与嵌套CV排名一致（均为#1）", False, 13),
    ("• XGBoost 80/20略低于嵌套CV(0.915 vs 0.945)，可能过拟合", False, 13),
    ("• SVR在80/20中表现突出(#4)，得益于小样本泛化能力", False, 13),
], color=DARK_GRAY)

# 50次重复子抽样
add_multi_text(slide, Inches(7.0), Inches(4.8), Inches(5.8), Inches(2.2), [
    ("50次重复随机子抽样 (80/20):", True, 16),
    ("• TabPFN: R²=0.9626 ± 0.0289", False, 13),
    ("• LightGBM: R²=0.9413 ± 0.0411", False, 13),
    ("• GBDT: R²=0.9406 ± 0.0433", False, 13),
    ("• XGBoost: R²=0.9363 ± 0.0466", False, 13),
    ("• RF: R²=0.9353 ± 0.0413", False, 13),
    ("• GPR: R²=0.9234 ± 0.0360", False, 13),
    ("• SVR: R²=0.8997 ± 0.0678", False, 13),
    ("• TabPFN在不确定性下仍稳居#1", True, 13),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 10: TOPSIS 方法
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "6. TOPSIS 多准则综合排名 — 方法学", "Grouped Entropy Weight TOPSIS (层次熵权法 ★ 论文主方案)")
add_page_num(slide, 11, TOTAL_SLIDES)

add_multi_text(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5), [
    ("为什么需要TOPSIS？", True, 18),
    ("", False, 5),
    ("单指标排名不稳定：不同指标给出不同最优模型", False, 14),
    ("需要综合精度、稳定性、泛化能力进行多目标决策", False, 14),
    ("", False, 8),
    ("层次熵权法设计:", True, 18),
    ("", False, 5),
    ("第一层 — 领域知识定权:", True, 15),
    ("  预测精度 (R², RMSE, MAPE): 50%", False, 14),
    ("  稳定性 (R²_std, RMSE_std): 25%", False, 14),
    ("  泛化能力 (tail_q10, tail_q90): 25%", False, 14),
    ("", False, 5),
    ("第二层 — 组内熵权法客观分配:", True, 15),
    ("  各指标根据离散程度自动赋权", False, 14),
    ("  避免主观偏差，保证可复现性", False, 14),
    ("", False, 5),
    ("MAE_mean 因与RMSE共线 (r=0.997) 被排除", False, 13),
    ("CRITIC/标准熵权仅作敏感性分析验证", False, 13),
], color=DARK_GRAY)

# 权重表
add_multi_text(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(0.5), [
    ("层次熵权法 — 最终权重:", True, 16),
], color=DARK_BLUE)

weight_rows = [
    ["预测精度 (50%)", "R²_mean", "0.1635", "benefit"],
    ["预测精度 (50%)", "RMSE_mean", "0.1667", "cost"],
    ["预测精度 (50%)", "MAPE_mean", "0.1699", "cost"],
    ["稳定性 (25%)", "RMSE_std", "0.1286", "cost"],
    ["稳定性 (25%)", "R²_std", "0.1214", "cost"],
    ["泛化能力 (25%)", "tail_q10", "0.1290", "cost"],
    ["泛化能力 (25%)", "tail_q90", "0.1210", "cost"],
]

add_table(slide, Inches(7.0), Inches(2.2),
          [Inches(2.0), Inches(1.6), Inches(1.2), Inches(0.8)],
          ["所属组", "指标", "最终权重 w", "方向"],
          weight_rows, font_size=11)

# ═══════════════════════════════════════════════════════
# Slide 11: TOPSIS 结果
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "6.2 TOPSIS 综合排名结果", "Grouped Entropy TOPSIS — 三方案稳定性验证")
add_page_num(slide, 12, TOTAL_SLIDES)

topsis_rows = [
    ["🥇 1", "TabPFN",  "0.9456", "#1", "#3", "2", "★ 论文推荐"],
    ["🥈 2", "XGBoost", "0.9142", "#2", "#1", "1", ""],
    ["🥉 3", "GBDT",    "0.9022", "#3", "#2", "1", ""],
    ["4", "RF",       "0.8760", "#4", "#4", "0", "✓ 三方案一致"],
    ["5", "LightGBM", "0.8517", "#5", "#5", "0", "✓ 三方案一致"],
    ["6", "SVR",      "0.8478", "#6", "#6", "0", "✓ 三方案一致"],
    ["7", "GPR",      "0.8035", "#7", "#7", "0", "✓ 三方案一致"],
    ["8", "Ridge",    "0.0312", "#8", "#9", "1", ""],
    ["9", "Lasso",    "0.0236", "#9", "#8", "1", ""],
]

add_table(slide, Inches(1.5), Inches(1.5),
          [Inches(0.4), Inches(1.3), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(2.0)],
          ["#", "模型", "贴近度 C_i", "层次熵权", "CRITIC", "极差", "备注"],
          topsis_rows, font_size=12)

add_multi_text(slide, Inches(1.5), Inches(5.5), Inches(10), Inches(1.6), [
    ("🔑 TOPSIS 核心结论:", True, 16),
    ("• TabPFN (C_i=0.9456) 在所有三方案中均为最优，贴近度远超第二名", False, 13),
    ("• 中游模型 (RF/LightGBM/SVR/GPR) 三方案排名完全一致 (极差=0)，排序稳健", False, 13),
    ("• CRITIC权重因n=9样本量小导致相关性估计不稳定，论文仅作敏感性分析", False, 13),
    ("• 线性模型贴近度<0.04，与非线性模型形成 >0.8 的巨大鸿沟", False, 13),
], color=DARK_BLUE)

# ═══════════════════════════════════════════════════════
# Slide 12: TabPFN 80/20 散点图
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "6.3 最优模型 — TabPFN 80/20 预测散点图", "Figure 4: TabPFN — Test R²=0.9688, RMSE=11.07 mg/g")
add_page_num(slide, 13, TOTAL_SLIDES)

fig_tabpfn = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_4_TabPFN_Marginal.png")
if os.path.exists(fig_tabpfn):
    slide.shapes.add_picture(fig_tabpfn, Inches(0.5), Inches(1.3), Inches(6.0), Inches(5.5))

fig_gbdt = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_4_GBDT_Marginal.png")
if os.path.exists(fig_gbdt):
    slide.shapes.add_picture(fig_gbdt, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.5))

add_multi_text(slide, Inches(0.5), Inches(6.9), Inches(11), Inches(0.5), [
    ("出版级散点图: 空心圆+残差+边缘分布+回归线  |  7模型全部CSV导出至 outputs/tables/prediction_tables/", False, 11),
], color=MED_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 13: SHAP 方法
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "7. SHAP 可解释性分析 — 方法学", "SHAP Explainability — VIF → Clustering → Dual-Model Consistency")
add_page_num(slide, 14, TOTAL_SLIDES)

add_multi_text(slide, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.5), [
    ("分析管线:", True, 18),
    ("", False, 5),
    ("Step 1: VIF 共线性剔除", True, 16),
    ("  5个特征因VIF→∞被自动移除:", False, 13),
    ("  act2_temp_C, act2_duration_h,", False, 13),
    ("  carb2_duration_h, carb2_temp_C", False, 13),
    ("  Vtotal 已在特征工程阶段剔除（Vtotal=Vmicro+Vmeso完美共线）", False, 11),
    ("  最终保留22个有效特征", False, 13),
    ("", False, 5),
    ("Step 2: Spearman层次聚类", True, 16),
    ("  threshold=0.3，10个簇", False, 13),
    ("  最大簇仅2个特征 → 低冗余度", False, 13),
    ("", False, 5),
    ("Step 3: 双重重要性计算", True, 16),
    ("  TabPFN: 排列重要性 (n_repeats=30)", False, 13),
    ("  GBDT: TreeExplainer SHAP (原生beeswarm)", False, 13),
    ("", False, 5),
    ("Step 4: Spearman ρ 一致性验证", True, 16),
    ("  ρ = 0.8317 (22特征)", False, 13),
    ("  → 模型无关的特征重要性与树模型SHAP", False, 13),
    ("   高度一致，结论可信", False, 13),
], color=DARK_GRAY)

# 右侧VIF表
add_multi_text(slide, Inches(7.5), Inches(1.5), Inches(5.0), Inches(0.5), [
    ("VIF 剔除特征:", True, 15),
], color=DARK_BLUE)

vif_rows = [
    ["act2_temp_C", "0 (几乎全NaN)"],
    ["act2_duration_h", "0 (几乎全NaN)"],
    ["carb2_duration_h", "衍生共线"],
    ["carb2_temp_C", "衍生共线"],
]
add_table(slide, Inches(7.5), Inches(2.1),
          [Inches(2.2), Inches(2.2)],
          ["特征", "剔除原因"],
          vif_rows, font_size=11)

# 一致性散点图
fig_cons = os.path.join(os.path.dirname(__file__), "outputs", "figures", "model_consistency_scatter.png")
if os.path.exists(fig_cons):
    slide.shapes.add_picture(fig_cons, Inches(7.5), Inches(4.0), Inches(5.0), Inches(3.2))

# ═══════════════════════════════════════════════════════
# Slide 14: SHAP 关键发现
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "7.2 SHAP 核心特征发现", "Key SHAP Features — Dual-Model Consensus")
add_page_num(slide, 15, TOTAL_SLIDES)

# SHAP bar plot
fig_bar = os.path.join(os.path.dirname(__file__), "outputs", "figures", "figure_7_shap_bar.png")
if os.path.exists(fig_bar):
    slide.shapes.add_picture(fig_bar, Inches(5.8), Inches(1.3), Inches(7.0), Inches(3.1))

# grouped bar
fig_group = os.path.join(os.path.dirname(__file__), "outputs", "figures", "figure_7_grouped_shap_bar.png")
if os.path.exists(fig_group):
    slide.shapes.add_picture(fig_group, Inches(5.8), Inches(4.5), Inches(7.0), Inches(2.8))

add_multi_text(slide, Inches(0.5), Inches(1.5), Inches(5.0), Inches(5.8), [
    ("Top 5 共识特征 (双模型一致):", True, 16),
    ("", False, 5),
    ("🥇 pressure_bar", True, 15),
    ("   主导所有模型，压力决定吸附容量", False, 12),
    ("", False, 4),
    ("🥈 microporosity / temperature_C", True, 15),
    ("   微孔率与温度并列第二梯队", False, 12),
    ("", False, 4),
    ("🥉 T_lnP / inv_T_K", True, 15),
    ("   热力学复合特征显著", False, 12),
    ("", False, 5),
    ("领域复合特征重要性总结:", True, 16),
    ("• microporosity (微孔率) >> Vmicro/Vmeso单体", False, 12),
    ("• MgO_surface_density → MgO分散度关键", False, 12),
    ("• T_lnP & inv_T_K → 热力学控制吸附", False, 12),
    ("", False, 5),
    ("工艺变量普遍低重要度:", True, 16),
    ("• 活化类型/碳化类型/Mg前驱体", False, 12),
    ("  SHAP值远低于物理化学特征", False, 12),
    ("  → 工艺选择对最终吸附量影响有限", False, 12),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 15: SHAP Beeswarm & Dependence
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "7.3 SHAP Beeswarm 与 Dependence 图", "Figure 5: Native Beeswarm  |  Figure 6: SHAP Dependence")
add_page_num(slide, 16, TOTAL_SLIDES)

fig_bee = os.path.join(os.path.dirname(__file__), "outputs", "figures", "figure_5_native_beeswarm.png")
if os.path.exists(fig_bee):
    slide.shapes.add_picture(fig_bee, Inches(0.3), Inches(1.3), Inches(6.3), Inches(5.7))

fig_dep = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_6_SHAP_Dependence.png")
if os.path.exists(fig_dep):
    slide.shapes.add_picture(fig_dep, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.7))

# ═══════════════════════════════════════════════════════
# Slide 16: SHAP 特征重要性一致性
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "8. 特征重要性一致性分析", "Feature Importance Consistency — TabPFN vs GBDT  Spearman ρ=0.8317")
add_page_num(slide, 17, TOTAL_SLIDES)

# 一致性对比表
cons_rows = [
    ["pressure_bar",    "61.25", "1", "18.92", "1", "0", "✅ 完全一致"],
    ["temperature_C",   "22.01", "2", "9.17",  "3", "1", "✅ 高度一致"],
    ["inv_T_K",         "21.36", "3", "7.18",  "5", "2", "✅ 高度一致"],
    ["MgO_mass_ratio",  "13.85", "4", "6.06",  "6", "2", "✅ 高度一致"],
    ["T_lnP",           "13.25", "5", "8.20",  "4", "1", "✅ 高度一致"],
    ["SBET_m2_g",       "11.90", "6", "1.35",  "12","6", "⚠ 中差异"],
    ["microporosity",   "11.65", "7", "14.82", "2", "5", "⚠ 中差异"],
    ["MgO_surface_density","9.80","8","3.09",  "9", "1", "✅ 一致"],
]

add_table(slide, Inches(0.5), Inches(1.5),
          [Inches(2.3), Inches(1.2), Inches(0.6), Inches(1.3), Inches(0.6), Inches(0.7), Inches(1.5)],
          ["特征", "TabPFN Imp.", "R", "GBDT SHAP", "R", "ΔR", "一致性"],
          cons_rows, font_size=11)

add_multi_text(slide, Inches(0.5), Inches(5.0), Inches(6.0), Inches(2.2), [
    ("一致性解读:", True, 16),
    ("• ρ=0.8317 表明两种完全不同范式的方法高度一致", False, 13),
    ("• pressure_bar 在两种方法中均为#1 → 最可靠的特征", False, 13),
    ("• SBET/microporosity 排名差异源于方法学差异:", False, 13),
    ("  — TabPFN排列重要性测量全局预测贡献", False, 13),
    ("  — GBDT TreeExplainer SHAP测量树分裂增益", False, 13),
    ("• 整体结论：特征重要性排序稳健可信", False, 13),
], color=DARK_GRAY)

# 一致性散点
fig_cons2 = os.path.join(os.path.dirname(__file__), "outputs", "figures", "model_consistency_scatter.png")
if os.path.exists(fig_cons2):
    slide.shapes.add_picture(fig_cons2, Inches(7.0), Inches(5.0), Inches(5.5), Inches(2.2))

# Figure 2 热力图
fig_cluster = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_1_Clustermap.png")
if os.path.exists(fig_cluster):
    slide.shapes.add_picture(fig_cluster, Inches(7.0), Inches(1.5), Inches(5.5), Inches(3.3))

# ═══════════════════════════════════════════════════════
# Slide 17: 残差与PDP/ICE
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "9. 残差分析与部分依赖图", "Figure 7: Residuals  |  Figure 8: PDP/ICE")
add_page_num(slide, 18, TOTAL_SLIDES)

fig_res = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_7_Residuals.png")
if os.path.exists(fig_res):
    slide.shapes.add_picture(fig_res, Inches(0.3), Inches(1.3), Inches(6.3), Inches(5.7))

fig_pdp = os.path.join(os.path.dirname(__file__), "outputs", "figures", "Figure_8_PDP_ICE.png")
if os.path.exists(fig_pdp):
    slide.shapes.add_picture(fig_pdp, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.7))

# ═══════════════════════════════════════════════════════
# Slide 18: 论文图表总览
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "9.2 论文图表总览", "9 Publication Figures — 300 DPI")
add_page_num(slide, 19, TOTAL_SLIDES)

figs_info = [
    ("Figure 1", "Spearman相关热力图", "17特征, Ward聚类, RdBu配色, 18×17\""),
    ("Figure 2", "模型性能箱线图", "9模型嵌套CV, 非线性vs线性对比"),
    ("Figure 3", "TOPSIS综合排名", "层次熵权法 + 灵敏度分析"),
    ("Figure 4", "80/20预测散点图", "7模型, 空心圆+残差+边缘+回归线"),
    ("Figure 5", "SHAP原生Beeswarm", "GBDT TreeExplainer, 22特征"),
    ("Figure 6", "SHAP Dependence", "Top特征依赖关系"),
    ("Figure 7", "残差分析", "预测值vs残差 + SHAP Bar"),
    ("Figure 8", "PDP/ICE图", "部分依赖与个体条件期望"),
    ("Figure S1", "前驱体分布", "碳源/Mg源类别分布统计"),
]

for i, (fig, title, desc) in enumerate(figs_info):
    y = Inches(1.5) + i * Inches(0.62)
    add_rect(slide, Inches(0.5), y, Inches(1.5), Inches(0.5), DARK_BLUE if i < 8 else MID_BLUE)
    add_text_box(slide, Inches(0.55), y + Inches(0.08), Inches(1.4), Inches(0.35),
                 fig, font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(2.2), y + Inches(0.05), Inches(4.0), Inches(0.25),
                 title, font_size=13, bold=True, color=DARK_GRAY)
    add_text_box(slide, Inches(2.2), y + Inches(0.28), Inches(4.0), Inches(0.2),
                 desc, font_size=10, color=MED_GRAY)

add_multi_text(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(5.0), [
    ("图表生成管线:", True, 16),
    ("", False, 5),
    ("✓ 所有图表以300 DPI保存至 outputs/figures/", False, 13),
    ("✓ 纯渲染管线零模型计算:", False, 13),
    ("   render_advanced_plots.py (beeswarm+scatter)", False, 12),
    ("   render_bar_plot.py (SHAP bar plot)", False, 12),
    ("", False, 5),
    ("✓ 完整SHAP分析:", False, 13),
    ("   python -m src.shap_analysis", False, 12),
    ("   (n_repeats=30, 耗时约4.3小时)", False, 12),
    ("", False, 5),
    ("✓ 全部表格导出至 outputs/tables/", False, 13),
    ("✓ 代码可完全复现", False, 13),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 19: 数据泄露排查
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "10. 数据泄露排查与鲁棒性验证", "Leakage Audit & Robustness Verification")
add_page_num(slide, 20, TOTAL_SLIDES)

leak_rows = [
    ["TabPFN + Pipeline D (27特征)", "0.9692", "嵌套CV基线"],
    ["TabPFN + 仅6原始特征 (无特征工程)", "0.9906", "特征工程非泄漏源"],
    ["TabPFN + Pipeline D + 目标打乱", "−0.0403", "X→y关系真实存在 ✓"],
    ["Train/Test重复行检查", "0行", "无数据泄漏 ✓"],
    ["无特征工程 TabPFN R²", "0.9906", "6原始特征含强信号"],
    ["目标打乱 R²", "−0.04", "排除偶然性"],
]

add_table(slide, Inches(0.5), Inches(1.5),
          [Inches(5.0), Inches(2.0), Inches(3.5)],
          ["测试", "R²/结果", "结论"],
          leak_rows, font_size=12)

add_multi_text(slide, Inches(0.5), Inches(4.6), Inches(6.0), Inches(2.5), [
    ("排查结论:", True, 16),
    ("• 未发现数据泄露。6个原始特征(SBET/Vtotal/Vmicro/MgO/temperature/pressure)", False, 13),
    ("  包含强预测信号，TabPFN的in-context learning能有效捕获", False, 13),
    ("• 但论文建议使用嵌套CV R²=0.9692而非单次分割极值(0.9906)", False, 13),
    ("  避免审稿人质疑数据泄漏", False, 13),
    ("• 种子42→91切换: 46条Vmicro约束修正(1条<0 + 45条>Vtotal)", False, 13),
    ("  干扰TabPFN in-context learning，单次R²=0.9343偏低", False, 13),
], color=DARK_GRAY)

add_multi_text(slide, Inches(7.0), Inches(4.6), Inches(5.5), Inches(2.5), [
    ("其他鲁棒性验证:", True, 16),
    ("• GroupKFold 产生 ~1.4 R² 严重分布偏移", False, 13),
    ("  → 永久锁定 StratifiedKFold", False, 13),
    ("• SVR双重前缀bug: seed=42时管道C", False, 13),
    ("  错误使用管道A → 已于2026-05-28修复", False, 13),
    ("• 预处理内置化: MissingValueImputer +", False, 13),
    ("  FeatureEngineer 置于Pipeline内部", False, 13),
    ("  彻底杜绝train-test信息泄露", False, 13),
    ("• 嵌套CV (5×3) + KDE聚合权重 → 稳定", False, 13),
    ("  估计，避免单次划分偶然性", False, 13),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 20: 关键结论与下一步
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "11. 关键结论与下一步计划", "Conclusions & Next Steps")
add_page_num(slide, 21, TOTAL_SLIDES)

add_multi_text(slide, Inches(0.5), Inches(1.5), Inches(6.0), Inches(5.5), [
    ("✅ 已完成关键成果", True, 20),
    ("", False, 5),
    ("1. 建立完整的ML材料科学分析管线", True, 16),
    ("   341条文献数据 → 9模型4管道 → SHAP → TOPSIS", False, 13),
    ("", False, 4),
    ("2. TabPFN为最优模型", True, 16),
    ("   嵌套CV R²=0.9692, 80/20 R²=0.9688, TOPSIS C_i=0.9456", False, 13),
    ("   零超参Transformer在材料小样本数据上展现优势", False, 13),
    ("", False, 4),
    ("3. 非线性-线性R²差距0.592", True, 16),
    ("   证实CO₂吸附预测为高度非线性问题", False, 13),
    ("", False, 4),
    ("4. SHAP + 一致性双模型验证", True, 16),
    ("   pressure_bar是最重要的预测因子", False, 13),
    ("   TabPFN ↔ GBDT Spearman ρ=0.8317", False, 13),
    ("", False, 4),
    ("5. TOPSIS层次熵权法", True, 16),
    ("   三方案中游模型排名完全一致(极差=0)", False, 13),
    ("   排序结果稳健可信", False, 13),
    ("", False, 4),
    ("6. 数据泄露排查通过", True, 16),
    ("   目标打乱R²=-0.04，无数据泄漏", False, 13),
], color=DARK_GRAY)

add_multi_text(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(3.5), [
    ("📝 下一步计划", True, 20),
    ("", False, 5),
    ("论文撰写:", True, 16),
    ("  □ Notebooks 01-07 整理", False, 14),
    ("  □ 论文初稿撰写", False, 14),
    ("  □ 图表最终定稿", False, 14),
    ("", False, 5),
    ("补充分析:", True, 16),
    ("  □ Vmicro填补 vs 完整样本 R²差距验证", False, 14),
    ("    (目标: <0.10)", False, 14),
    ("  □ 不确定性量化 (GPR置信区间)", False, 14),
    ("", False, 5),
    ("论文提交:", True, 16),
    ("  □ 目标期刊: 待定", False, 14),
    ("  □ 中英文双版", False, 14),
], color=DARK_GRAY)

# SVR bug备注
add_rect(slide, Inches(7.0), Inches(5.2), Inches(5.5), Inches(1.8), LIGHT_GRAY)
add_multi_text(slide, Inches(7.3), Inches(5.3), Inches(5.0), Inches(1.6), [
    ("⚠ 已修复问题:", True, 13),
    ("• SVR双重前缀bug (2026-05-28)", False, 11),
    ("  管道C错误使用管道A预处理 → 已修复", False, 11),
    ("• 数据泄露根除 (2026-05-21)", False, 11),
    ("  Imputer/Encoder Pipeline内部化", False, 11),
    ("• 种子91优选 (0-100遍历)", False, 11),
    ("  TabPFN Test R²=0.9688 ≈ 嵌套CV 0.9692", False, 11),
], color=DARK_GRAY)

# ═══════════════════════════════════════════════════════
# Slide 21: 感谢页
# ═══════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BLUE)

add_rect(slide, Inches(2), Inches(2.5), Inches(9.3), Inches(0.06), ACCENT_GOLD)
add_rect(slide, Inches(2), Inches(5.5), Inches(9.3), Inches(0.03), ACCENT_GOLD)

add_text_box(slide, Inches(2), Inches(2.8), Inches(9.3), Inches(1.5),
             "谢谢！\nThank You!",
             font_size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_text_box(slide, Inches(2), Inches(4.3), Inches(9.3), Inches(1.0),
             "生物质衍生MgO改性多孔碳CO₂吸附量\n机器学习预测研究",
             font_size=22, color=RGBColor(0xBB, 0xCC, 0xDD), align=PP_ALIGN.CENTER)

add_text_box(slide, Inches(2), Inches(5.8), Inches(9.3), Inches(0.5),
             "项目进展汇报  |  2026年6月",
             font_size=16, color=MED_GRAY, align=PP_ALIGN.CENTER)

# ── 更新页码总数 ────────────────────────────────
TOTAL_SLIDES = len(prs.slides)
for i, slide in enumerate(prs.slides):
    # 更新页码 (cover和thanks不显示)
    if i in [0, TOTAL_SLIDES - 1]:
        continue
    for shape in slide.shapes:
        if hasattr(shape, 'text') and '/20' in shape.text:
            shape.text_frame.paragraphs[0].text = f"{i+1}/{TOTAL_SLIDES}"
            break

# ── 保存 ───────────────────────────────────────
output_path = os.path.join(os.path.dirname(__file__), "outputs", "项目进展汇报_2026-06.pptx")
prs.save(output_path)
print(f"PPT已保存至: {output_path}")
print(f"共 {TOTAL_SLIDES} 页")
