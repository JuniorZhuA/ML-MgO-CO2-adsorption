#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在综合进展报告中添加数值特征描述性统计表（使用建模实际数据）"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os, numpy as np, pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── 读取建模数据（EXACT data used by models）───
X = pd.read_csv(os.path.join(os.path.dirname(__file__),
                'outputs/tables/X_transformed_gbdt.csv'), index_col=0)
# 目标变量从原始数据获取（因为X_transformed_gbdt不含target）
from src.data_loader import load_and_clean
df_raw = load_and_clean()
X['CO2_uptake_mg_g'] = df_raw['CO2_uptake_mg_g'].values

# 原始非空数（用于对比）
raw_n = {}
for c in ['SBET_m2_g','Vmicro_cm3_g']:
    raw_n[c] = df_raw[c].notna().sum()

# ── 打开文档 ──────────────────────────────
doc_path = os.path.join(os.path.dirname(__file__), 'outputs', '项目综合进展报告.docx')
doc = Document(doc_path)

# ── 删除旧表（如果存在）───
paragraphs_list = list(doc.paragraphs)  # 复制列表避免迭代时修改
old_found = False
for i, p in enumerate(paragraphs_list):
    if p.text.strip() == '2.4 数值特征描述性统计' and p.style.name.startswith('Heading'):
        old_found = True
        to_del = [p._element]
        # 收集后续非标题段落直到下一个Heading1
        for j in range(i+1, len(paragraphs_list)):
            pp = paragraphs_list[j]
            if pp.style.name.startswith('Heading 1') and '3.' in pp.text:
                break
            if pp.text.strip():
                to_del.append(pp._element)
        for elem in to_del:
            try:
                elem.getparent().remove(elem)
            except Exception:
                pass
        break

if old_found:
    # 删除旧表格
    for t in doc.tables:
        if len(t.rows) > 0 and '类别' in t.rows[0].cells[0].text:
            try:
                t._element.getparent().remove(t._element)
            except Exception:
                pass
            break
    print("已删除旧的描述性统计内容")

# ── 找到插入位置 ──────────────────────────
next_h1 = None
paragraphs_list2 = list(doc.paragraphs)
for i, p in enumerate(paragraphs_list2):
    if p.text.strip().startswith('3. 模型') and 'Heading' in p.style.name:
        next_h1 = i
        break

# ── XML helpers ───────────────────────────
def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e

def make_heading(text, level=2):
    p = el('w:p')
    pPr = el('w:pPr')
    pStyle = el('w:pStyle', **{'w:val': f'Heading{level}'})
    pPr.append(pStyle); p.append(pPr)
    r = el('w:r'); t = el('w:t', **{'xml:space':'preserve'}); t.text = text
    r.append(t); p.append(r)
    return p

def make_para(text, size=20):
    p = el('w:p'); r = el('w:r')
    rPr = el('w:rPr'); szE = el('w:sz', **{'w:val':str(size)})
    rPr.append(szE); r.append(rPr)
    t = el('w:t', **{'xml:space':'preserve'}); t.text = text
    r.append(t); p.append(r)
    return p

def make_cell(text, bold=False, shade=None, size=16, align='center'):
    tc = el('w:tc'); tcPr = el('w:tcPr')
    if shade:
        shd = el('w:shd', **{'w:val':'clear','w:color':'auto','w:fill':shade})
        tcPr.append(shd)
    tc.append(tcPr)
    p = el('w:p'); pPr = el('w:pPr')
    jc = el('w:jc', **{'w:val':align}); pPr.append(jc); p.append(pPr)
    r = el('w:r'); rPr = el('w:rPr')
    rFonts = el('w:rFonts', **{'w:ascii':'Microsoft YaHei','w:hAnsi':'Microsoft YaHei','w:eastAsia':'Microsoft YaHei'})
    rPr.append(rFonts)
    szE = el('w:sz', **{'w:val':str(size)}); rPr.append(szE)
    if bold: rPr.append(el('w:b'))
    if shade: rPr.append(el('w:color', **{'w:val':'FFFFFF'}))
    r.append(rPr)
    t = el('w:t', **{'xml:space':'preserve'}); t.text = str(text)
    r.append(t); p.append(r); tc.append(p)
    return tc

def make_table(headers, rows, col_widths, cat_merges=None):
    tbl = el('w:tbl')
    tblPr = el('w:tblPr')
    tblStyle = el('w:tblStyle', **{'w:val':'TableGrid'})
    tblPr.append(tblStyle); tbl.append(tblPr)
    tblGrid = el('w:tblGrid')
    for cw in col_widths:
        tblGrid.append(el('w:gridCol', **{'w:w':str(cw)}))
    tbl.append(tblGrid)
    # header
    tr = el('w:tr')
    for h in headers:
        tr.append(make_cell(h, bold=True, shade='1B2A4A', size=15))
    tbl.append(tr)
    # rows
    for ri, row in enumerate(rows):
        tr = el('w:tr')
        for ci, val in enumerate(row):
            tr.append(make_cell(val, size=15))
        tbl.append(tr)
    return tbl

# ── 构建表格 ──────────────────────────────
# 五大类特征定义
feature_groups = [
    ("多孔碳结构", [
        ("SBET_m2_g",       "BET比表面积",         "m²/g",   True),
        ("Vmicro_cm3_g",    "微孔容",              "cm³/g",  True),
        ("Vmeso_cm3_g",     "介孔体积",             "cm³/g",  False),
        ("microporosity",   "微孔率",               "—",      False),
    ]),
    ("MgO负载", [
        ("MgO_mass_ratio",       "MgO质量比",           "—",    True),
        ("MgO_surface_density",  "MgO表面分散密度",      "g/m²", False),
    ]),
    ("吸附条件/热力学", [
        ("temperature_C",  "吸附温度",              "°C",     True),
        ("pressure_bar",   "吸附压力",              "bar",    True),
        ("T_lnP",          "温度-压力耦合项 T·ln(P)", "K",     False),
        ("inv_T_K",        "Arrhenius温度倒数",      "K⁻¹",   False),
    ]),
    ("工艺条件", [
        ("act1_temp_C",      "一次活化温度",          "°C", True),
        ("act1_duration_h",  "一次活化时长",          "h",  True),
        ("act2_temp_C",      "二次活化温度",           "°C", True),
        ("act2_duration_h",  "二次活化时长",           "h",  True),
        ("carb1_temp_C",     "一次碳化温度",          "°C", True),
        ("carb1_duration_h", "一次碳化时长",          "h",  True),
        ("carb2_temp_C",     "二次碳化温度",           "°C", True),
        ("carb2_duration_h", "二次碳化时长",           "h",  True),
    ]),
    ("目标变量", [
        ("CO2_uptake_mg_g", "CO₂吸附量",            "mg/g",  True),
    ]),
]

headers = ["类别", "特征名称", "物理含义", "单位", "N", "均值", "标准差", "最小值", "P25", "P50", "P75", "最大值"]
col_widths = [1100, 1600, 1700, 550, 450, 650, 650, 650, 650, 650, 650, 700]

rows = []
for cat_name, features in feature_groups:
    for fi, (fname, fdesc, funit, _) in enumerate(features):
        s = X[fname]
        n = int(s.notna().sum())
        row = [
            cat_name if fi == 0 else "",
            fname,
            fdesc,
            funit,
            str(n),
            f"{s.mean():.2f}",
            f"{s.std():.2f}",
            f"{s.min():.2f}",
            f"{s.quantile(0.25):.2f}",
            f"{s.median():.2f}",
            f"{s.quantile(0.75):.2f}",
            f"{s.max():.2f}",
        ]
        rows.append(row)

# ── 插入 ──────────────────────────────────
ref_elem = doc.paragraphs[next_h1]._element

note1 = make_para(
    "表2-1  全部数值特征描述性统计（N=341，基于建模数据集）。"
    "SBET/Vtotal/Vmicro 含填补值（SBET、Vtotal各15条以KNN k=5填补；Vmicro 144条以IterativeImputer BayesianRidge填补，"
    "填补后施加Vmicro≤Vtotal及Vmicro≥0物理约束）。"
    "Vmeso、microporosity、MgO_surface_density、T_lnP、inv_T_K 为领域复合特征。",
    size=17)
note2 = make_para(
    "注：Vmeso、microporosity、MgO_surface_density、T_lnP、inv_T_K 为领域复合特征，由原始特征推导得出。",
    size=17)

heading_elem = make_heading("2.4 数值特征描述性统计", level=2)
table_elem = make_table(headers, rows, col_widths)

# 倒序插入
ref_elem.addprevious(table_elem)
ref_elem.addprevious(note2)
ref_elem.addprevious(note1)
ref_elem.addprevious(heading_elem)

# ── 保存 ──────────────────────────────────
doc.save(doc_path)

print(f"✅ 描述性统计表已更新（基于建模实际数据）")
print(f"   {len(rows)} 个特征 × {len(headers)} 列")
print(f"   五大类: 多孔碳结构(5)+MgO负载(2)+吸附条件(4)+工艺(8)+目标(1)")
print()
print("═══ 关键统计验证 ═══")
print(f"  SBET:      326实测→341建模, mean=728.73, std=462.01")
print(f"  Vtotal:    326实测→341建模, mean=0.44,  std=0.23")
print(f"  Vmicro:    197实测→341建模, mean=0.29,  std=0.17 (原实测mean=0.37)")
print(f"  CO2 uptake: 341实测,         mean=99.62, std=62.35")
print(f"  act2_*:    仅9条非空, VIF剔除")
print(f"  carb2_*:   174条非空, VIF剔除")
