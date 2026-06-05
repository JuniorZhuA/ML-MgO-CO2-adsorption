"""将 manuscript_section_2_data_and_methodology.md 转换为格式化的 Word 文档。"""

import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import docx.oxml

# ── 文档设置 ──────────────────────────────────────────────────────
doc = Document()

# 页面设置
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

# 默认字体
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

# 标题样式
for level, (size, bold, space_before) in enumerate([
    (16, True, 18),  # Heading 1
    (14, True, 14),  # Heading 2
    (13, True, 10),  # Heading 3
], start=1):
    h_style = doc.styles[f'Heading {level}']
    h_font = h_style.font
    h_font.name = 'Times New Roman'
    h_font.size = Pt(size)
    h_font.bold = bold
    h_font.color.rgb = RGBColor(0, 0, 0)
    h_style.paragraph_format.space_before = Pt(space_before)
    h_style.paragraph_format.space_after = Pt(6)
    h_style.paragraph_format.line_spacing = 1.15

# ── 读取 Markdown ─────────────────────────────────────────────────
md_path = "outputs/manuscript_section_2_data_and_methodology.md"
with open(md_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ── 状态机解析 ────────────────────────────────────────────────────
def parse_inline(text, paragraph):
    """解析行内格式: **bold**, *italic*, `code`, $$math$$, $math$"""
    # 先处理 $$...$$ (display math)
    parts = re.split(r'(\$\$.*?\$\$)', text)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            math_text = part[2:-2]
            run = paragraph.add_run(math_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.italic = True
        else:
            parse_inline_simple(part, paragraph)

def parse_inline_simple(text, paragraph):
    """处理 $math$, **bold**, *italic*, `code`"""
    # $...$ math
    parts = re.split(r'(\$.*?\$)', text)
    for part in parts:
        if part.startswith('$') and part.endswith('$') and not part.startswith('$$'):
            math_text = part[1:-1]
            run = paragraph.add_run(math_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.italic = True
        else:
            parse_formatting(part, paragraph)

def parse_formatting(text, paragraph):
    """处理 **bold**, *italic*, `code`"""
    # 按 **...** 分割
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            bold_text = part[2:-2]
            # 检查内部是否有 *italic*
            sub_parts = re.split(r'(\*.*?\*)', bold_text)
            for sp in sub_parts:
                if sp.startswith('*') and sp.endswith('*'):
                    run = paragraph.add_run(sp[1:-1])
                    run.bold = True
                    run.italic = True
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                else:
                    parse_code(sp, paragraph, bold=True)
        else:
            # 检查 *italic*
            sub_parts = re.split(r'(\*[^*].*?\*)', part)
            for sp in sub_parts:
                if sp.startswith('*') and sp.endswith('*') and not sp.startswith('**'):
                    parse_code(sp[1:-1], paragraph, italic=True)
                else:
                    parse_code(sp, paragraph)

def parse_code(text, paragraph, bold=False, italic=False):
    """处理 `code`"""
    parts = re.split(r'(`.*?`)', text)
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(10.5)
            run.bold = bold
            run.italic = italic
        else:
            if part:
                run = paragraph.add_run(part)
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.bold = bold
                run.italic = italic

def add_formatted_paragraph(doc, text, style_name='Normal', bold=False, italic=False, alignment=None):
    """添加一个格式化段落。"""
    p = doc.add_paragraph(style=style_name)
    parse_inline(text, p)
    if alignment is not None:
        p.alignment = alignment
    return p

def add_bullet(doc, text, level=0):
    """添加列表项。"""
    p = doc.add_paragraph(style='List Bullet')
    parse_inline(text, p)
    if level > 0:
        p.paragraph_format.left_indent = Cm(1.27 * (level + 1))
    return p

def add_table_from_markdown(doc, headers, rows):
    """从表头和数据行创建格式化的 Word 表格。"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(h.strip())
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # 灰色背景
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="D9E2F3"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    # 数据行
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(val).strip())
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)

    # 列宽自动调整
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(1)
                paragraph.paragraph_format.space_after = Pt(1)

    doc.add_paragraph()  # 表后空行
    return table

# ── 逐行解析 ──────────────────────────────────────────────────────
i = 0
code_block_lines = []
in_code_block = False
in_md_table = False
table_lines = []

while i < len(lines):
    line = lines[i].rstrip()

    # 代码块处理
    if line.startswith('```'):
        if in_code_block:
            # 代码块结束
            code_text = '\n'.join(code_block_lines)
            p = doc.add_paragraph()
            run = p.add_run(code_text)
            run.font.name = 'Consolas'
            run.font.size = Pt(9)
            p.paragraph_format.left_indent = Cm(1)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            # 浅灰背景
            pPr = p._p.get_or_add_pPr()
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:fill="F2F2F2"/>')
            pPr.append(shading)
            code_block_lines = []
            in_code_block = False
        else:
            in_code_block = True
        i += 1
        continue

    if in_code_block:
        code_block_lines.append(line)
        i += 1
        continue

    # 表格处理
    if line.startswith('|') and line.strip().endswith('|'):
        # 跳过表头分隔行 (|---|...|)
        if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
            table_lines.append(line)  # 保存分隔行用于后续处理
            i += 1
            continue
        table_lines.append(line)
        # 看看下一行是否还是表格
        if i + 1 < len(lines) and lines[i + 1].strip().startswith('|'):
            i += 1
            continue
        else:
            # 表格结束，解析
            if table_lines:
                # 第一行是表头，可能存在分隔行
                header_line = table_lines[0]
                headers = [c.strip() for c in header_line.strip('|').split('|')]

                # 找到数据行（跳过表头分隔行）
                data_lines = []
                for tl in table_lines[1:]:
                    if re.match(r'^\|[\s\-:|]+\|$', tl.strip()):
                        continue
                    row = [c.strip() for c in tl.strip('|').split('|')]
                    data_lines.append(row)

                if data_lines:
                    add_table_from_markdown(doc, headers, data_lines)
            table_lines = []
            i += 1
            continue

    # 空行
    if not line.strip():
        table_lines = []
        i += 1
        continue

    # 标题
    if line.startswith('# '):
        add_formatted_paragraph(doc, line[2:], 'Heading 1', bold=True)
    elif line.startswith('## '):
        add_formatted_paragraph(doc, line[3:], 'Heading 2', bold=True)
    elif line.startswith('### '):
        add_formatted_paragraph(doc, line[4:], 'Heading 3', bold=True)
    elif line.startswith('#### '):
        p = doc.add_paragraph()
        parse_inline(f'**{line[5:]}**', p)
        p.runs[0].font.size = Pt(12)
        p.paragraph_format.space_before = Pt(8)

    # 列表项
    elif re.match(r'^\d+\.\s+\*\*', line):  # 1. **bold text**...
        text = re.sub(r'^\d+\.\s+', '', line)
        add_bullet(doc, text)
    elif re.match(r'^\d+\.\s+', line):
        add_bullet(doc, re.sub(r'^\d+\.\s+', '', line))
    elif re.match(r'^[\-\*]\s+', line):
        add_bullet(doc, re.sub(r'^[\-\*]\s+', '', line))

    # 块引用 (> ...)
    elif line.startswith('> '):
        text = line[2:]
        if text.startswith('**Note.**'):
            p = doc.add_paragraph()
            parse_inline(text, p)
            p.paragraph_format.left_indent = Cm(1)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            # 为 "Note." 部分加粗
            if p.runs:
                p.runs[0].bold = True

    # 普通段落
    else:
        p = doc.add_paragraph()
        parse_inline(line, p)

    i += 1

# ── 最终排版调整 ──────────────────────────────────────────────────

# 所有表格单元格字体统一为 Times New Roman
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    if run.font.name == 'Calibri' or run.font.name is None:
                        run.font.name = 'Times New Roman'

# ── 保存 ──────────────────────────────────────────────────────────
output_path = "outputs/Section_2_Data_and_Methodology.docx"
doc.save(output_path)
print(f"✓ Word 文档已生成: {output_path}")
print(f"  页面设置: 上下左右 2.54cm, Times New Roman 12pt")
print(f"  表格数: {len(doc.tables)}")
print(f"  段落数: {len(doc.paragraphs)}")
