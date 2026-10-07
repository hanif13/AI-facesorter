"""Thai local UI for finding, reviewing and exporting event photos by person.

Premium black-and-white design with Inter font, responsive layout,
and a clear 3-step flow: Upload → Process → Download.
"""
from __future__ import annotations
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st
from src.common.image_io import image_paths, load_rgb
from src.folder_picker import choose_folder
from src.downloads import prepare_zip, LocalDownloads
from src.product import (ROOT, PROJECTS, DEFAULT_SOURCE, read_json, write_json, project_path,
                         catalog, edit_project, export_project, recluster_project)

# ─── page config ───
st.set_page_config(page_title='Face Sorter • จัดรูปตามคน', page_icon='◐', layout='wide')

# ─── premium CSS ───
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Noto+Sans+Thai:wght@300;400;500;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown('''<style>

/* ── Root variables ── */
:root {
    --bg: #ffffff;
    --bg-elevated: #fafafa;
    --bg-card: #ffffff;
    --border: #e5e5e5;
    --border-light: #f0f0f0;
    --text-primary: #0a0a0a;
    --text-secondary: #525252;
    --text-muted: #a3a3a3;
    --accent: #000000;
    --accent-hover: #262626;
    --radius: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,.06);
    --shadow-lg: 0 8px 30px rgba(0,0,0,.08);
    --transition: all .2s cubic-bezier(.4,0,.2,1);
}

/* ── Global typography ── */
html, body, .stApp, .stMarkdown, p, label,
button, input, select, textarea {
    font-family: 'Inter', 'Noto Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}



/* ── Main container ── */
.stApp { background: var(--bg); }
.block-container {
    max-width: 1200px;
    padding: 2rem 1.5rem 4rem !important;
}

/* ── Title styling ── */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    font-size: clamp(1.75rem, 4vw, 2.5rem) !important;
    color: var(--text-primary) !important;
    line-height: 1.15 !important;
    margin-bottom: .25rem !important;
}
h2, .stSubheader {
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    font-size: clamp(1.15rem, 2.5vw, 1.5rem) !important;
    color: var(--text-primary) !important;
}
h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stTitle {
    font-size: 1.25rem !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.25rem 1rem !important;
    transition: var(--transition);
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-md);
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    font-size: 1.75rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 500 !important;
    font-size: .8rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: .04em !important;
}

/* ── Step indicator badges ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .01em;
    transition: var(--transition);
}
.step-badge.active {
    background: var(--accent);
    color: #fff;
}
.step-badge.done {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border);
}
.step-badge.pending {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border-light);
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    font-size: .7rem;
    font-weight: 700;
}
.step-badge.active .step-num { background: rgba(255,255,255,.2); color: #fff; }
.step-badge.done .step-num { background: var(--accent); color: #fff; }
.step-badge.pending .step-num { background: var(--border-light); color: var(--text-muted); }

/* ── Step flow container ── */
.step-flow {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin: 1rem 0 1.5rem;
}
.step-connector {
    width: 24px;
    height: 1px;
    background: var(--border);
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    letter-spacing: .01em !important;
    padding: .6rem 1.5rem !important;
    transition: var(--transition) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg) !important;
    color: var(--text-primary) !important;
}
.stButton > button:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
}
.stFormSubmitButton > button:hover,
.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px);
}

/* ── Form & Inputs ── */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    font-size: .85rem !important;
    padding: .6rem .875rem !important;
    transition: var(--transition);
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,.06) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--bg-elevated);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border-light);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: .85rem !important;
    padding: .5rem 1.25rem !important;
    color: var(--text-secondary) !important;
    transition: var(--transition);
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: .85rem !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: var(--bg-elevated) !important;
    border-radius: 100px !important;
    height: 6px !important;
}
.stProgress > div > div > div {
    background: var(--accent) !important;
    border-radius: 100px !important;
}

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-light) !important;
    margin: 1.5rem 0 !important;
}

/* ── Caption / muted text ── */
.stCaption, [data-testid="stCaption"] {
    color: var(--text-muted) !important;
    font-size: .78rem !important;
    line-height: 1.5 !important;
}

/* ── Image cards ── */
[data-testid="stImage"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
    border: 1px solid var(--border-light);
    transition: var(--transition);
}
[data-testid="stImage"]:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border);
}

/* ── Checkbox ── */
.stCheckbox label span {
    font-size: .82rem !important;
    font-weight: 500 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
}

/* ── Privacy badge ── */
.privacy-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-light);
    font-size: .72rem;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: .02em;
}

/* ── Hero section ── */
.hero-subtitle {
    font-size: clamp(.85rem, 1.5vw, 1rem);
    color: var(--text-secondary);
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
    margin-bottom: 1.5rem;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem .75rem 3rem !important;
    }
    [data-testid="stMetric"] {
        padding: .875rem .75rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }
    .step-flow {
        gap: 4px;
    }
    .step-connector {
        width: 12px;
    }
    .step-badge {
        padding: 4px 10px;
        font-size: .72rem;
    }
}

@media (max-width: 480px) {
    h1 {
        font-size: 1.5rem !important;
    }
    .block-container {
        padding: .75rem .5rem 2rem !important;
    }
    .step-flow {
        flex-direction: column;
        align-items: flex-start;
    }
    .step-connector {
        display: none;
    }
}
</style>''', unsafe_allow_html=True)


# ─── helpers ───
@st.cache_data(max_entries=300)
def thumbnail(path: str, modified: int, size=240):
    image = load_rgb(Path(path))
    image.thumbnail((size, size))
    data = io.BytesIO()
    image.save(data, format='JPEG', quality=85)
    return data.getvalue()


def show_image(path, size=240):
    p = Path(path)
    try:
        st.image(thumbnail(str(p), p.stat().st_mtime_ns, size), width='stretch')
    except (OSError, ValueError):
        st.caption('ไม่พบรูปต้นฉบับ')


def step_indicator(current: int):
    """Render step badges: 1=Upload  2=Process  3=Download."""
    labels = ['อัปโหลด', 'ประมวลผล', 'ดาวน์โหลด']
    icons = ['↑', '⚙', '↓']
    parts = []
    for i, (label, icon) in enumerate(zip(labels, icons), start=1):
        if i < current:
            cls = 'done'
        elif i == current:
            cls = 'active'
        else:
            cls = 'pending'
        parts.append(f'<span class="step-badge {cls}"><span class="step-num">{i}</span>{icon} {label}</span>')
        if i < 3:
            parts.append('<span class="step-connector"></span>')
    st.markdown(f'<div class="step-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


# ─── sidebar ───
PROJECTS.mkdir(parents=True, exist_ok=True)
projects = sorted(p for p in PROJECTS.iterdir() if p.is_dir() and ((p/'project.json').exists() or (p/'status.json').exists()))

with st.sidebar:
    st.markdown('### ◐ Face Sorter')
    st.caption('จัดรูปงานตามบุคคล อัตโนมัติ')
    st.markdown('---')

    choices = ['＋ สร้างชุดรูปใหม่'] + [p.name for p in projects]
    pending = st.session_state.pop('pending_project', None)
    if pending in choices:
        st.session_state.project_choice = pending
    default_project = 'day3_ready' if 'day3_ready' in choices else ('day3_product' if 'day3_product' in choices else choices[0])
    if st.session_state.get('project_choice') not in choices:
        st.session_state.project_choice = default_project
    chosen = st.selectbox('ชุดรูป', choices, key='project_choice', label_visibility='collapsed')

    st.markdown('---')
    st.markdown('''
<div style="display:flex;flex-direction:column;gap:12px;padding:8px 0;">
<div style="display:flex;align-items:center;gap:10px;">
    <span style="display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.7rem;font-weight:700;">1</span>
    <span style="font-size:.82rem;font-weight:500;color:#525252;">เลือกโฟลเดอร์รูป</span>
</div>
<div style="display:flex;align-items:center;gap:10px;">
    <span style="display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.7rem;font-weight:700;">2</span>
    <span style="font-size:.82rem;font-weight:500;color:#525252;">ตรวจและแก้กลุ่มบุคคล</span>
</div>
<div style="display:flex;align-items:center;gap:10px;">
    <span style="display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.7rem;font-weight:700;">3</span>
    <span style="font-size:.82rem;font-weight:500;color:#525252;">ส่งออกรูปที่ต้องการ</span>
</div>
</div>
''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<div class="privacy-badge">🔒 ทำงานในเครื่อง · ไม่อัปโหลดข้อมูล</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — UPLOAD / SELECT SOURCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if chosen == '＋ สร้างชุดรูปใหม่':
    st.title('จัดรูปงาน ตามคนที่อยู่ในภาพ')
    st.markdown('<p class="hero-subtitle">เลือกโฟลเดอร์รูป ระบบจะค้นหาใบหน้า จัดกลุ่มคนเดียวกัน แล้วส่งออกเป็นโฟลเดอร์แยกตามบุคคล</p>', unsafe_allow_html=True)

    step_indicator(1)

    def select_source_folder():
        try:
            previous = st.session_state.get('source_folder')
            selected = choose_folder(Path(previous or DEFAULT_SOURCE))
            if selected is not None:
                st.session_state.source_folder = str(selected)
            st.session_state.pop('folder_picker_error', None)
        except RuntimeError as error:
            st.session_state.folder_picker_error = str(error)

    st.button('เลือกโฟลเดอร์รูป', on_click=select_source_folder, type='primary',
              help='เปิดหน้าต่างเลือกโฟลเดอร์บนเครื่องนี้ รองรับ JPG, PNG, HEIC และ WebP')
    if st.session_state.get('folder_picker_error'):
        st.error(st.session_state.folder_picker_error)
    source_text = st.session_state.get('source_folder')
    with st.form('new_project'):
        st.markdown('##### เลือกรูปต้นฉบับ')
        if source_text:
            st.success(f'เลือกแล้ว: {Path(source_text).name}')
            st.caption('ใช้รูปจากโฟลเดอร์ที่เลือกโดยตรง รวมรูปในโฟลเดอร์ย่อย')
        else:
            st.info('กดเลือกโฟลเดอร์รูปด้านบนเพื่อเริ่มต้น')
        name = st.text_input(
            'ชื่อชุดรูป',
            'งานใหม่_' + datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%m%d_%H%M'),
            help='ตั้งชื่อเพื่อจำแนกชุดรูปแต่ละงาน'
        )

        st.markdown('')
        with st.expander('⚙ ตัวเลือกขั้นสูง'):
            alg = st.selectbox('วิธีจัดกลุ่ม', ['dbscan', 'hdbscan', 'agglomerative'])
            threshold = st.slider('ระยะห่างสูงสุด (DBSCAN / Agglomerative)', .05, .9, .4, .01)
            rescue = st.checkbox('นำใบหน้าที่ยังไม่มีกลุ่มเข้ากลุ่มที่คล้ายกัน', False)
            similarity = st.slider('ความคล้ายขั้นต่ำสำหรับนำเข้ากลุ่ม', .2, .9, .45, .01)
            st.caption('ค่าต่ำอาจรวมคนละคน · ค่าสูงอาจแยกคนเดียวกันเป็นหลายกลุ่ม')

        st.markdown('')
        submitted = st.form_submit_button('เริ่มค้นหาและจัดกลุ่มใบหน้า ▸', type='primary', disabled=not source_text)

    if submitted:
        try:
            source = Path(source_text).expanduser().resolve()
            if not source.is_dir() or not image_paths(source):
                raise ValueError('ไม่พบโฟลเดอร์รูป หรือโฟลเดอร์ไม่มีรูปที่รองรับ')
            project = project_path(name)
            if project.exists():
                raise ValueError('ชื่อชุดรูปนี้มีอยู่แล้ว กรุณาเปิดจากเมนูหรือใช้ชื่อใหม่')
            project.mkdir(parents=True)
            write_json(project / 'status.json', {'state': 'running', 'phase': 'กำลังเริ่มงาน', 'done': 0, 'total': 0})
            command = [sys.executable, str(ROOT / 'run_pipeline.py'), '--input', str(source), '--run-name', name,
                       '--algo', alg, '--threshold', str(threshold), '--rescue-sim', str(similarity)]
            if not rescue:
                command.append('--no-rescue')
            with (project / 'console.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            st.session_state[f'job_{name}'] = proc
            st.session_state['pending_project'] = name
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — PROCESSING (progress)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
project = project_path(chosen)
status = read_json(project / 'status.json', {'state': 'running', 'phase': 'กำลังเริ่มงาน'})

if status.get('state') != 'ready':
    st.title('กำลังประมวลผล')
    st.markdown(f'<p class="hero-subtitle">ชุดรูป <strong>{chosen}</strong> — ระบบกำลังค้นหาและจัดกลุ่มใบหน้า</p>', unsafe_allow_html=True)

    step_indicator(2)

    @st.fragment(run_every='3s')
    def progress_panel():
        current = read_json(project / 'status.json', status)
        proc = st.session_state.get(f'job_{chosen}')
        if proc and proc.poll() is not None and current.get('state') == 'running':
            current = {'state': 'failed', 'error': 'งานหยุดก่อนเสร็จ กรุณาลองทำต่อ'}
        if current.get('state') == 'ready':
            st.rerun()
        elif current.get('state') == 'failed':
            st.error(current.get('error', 'ประมวลผลไม่สำเร็จ'))
            if st.button('ลองทำต่อจากขั้นตอนที่เสร็จแล้ว'):
                config = read_json(project / 'project.json')
                if config and config.get('source'):
                    command = [sys.executable, str(ROOT / 'run_pipeline.py'), '--input', config['source'], '--run-name', chosen,
                               '--algo', config.get('algorithm', 'hdbscan'), '--threshold', str(config.get('threshold', .45))]
                    if config.get('rescue_sim', .45) is None:
                        command.append('--no-rescue')
                    else:
                        command += ['--rescue-sim', str(config.get('rescue_sim', .45))]
                    if config.get('sample_limit'):
                        command += ['--limit', str(config['sample_limit'])]
                    with (project / 'console.log').open('a') as log:
                        st.session_state[f'job_{chosen}'] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    write_json(project / 'status.json', {'state': 'running', 'phase': 'กำลังทำต่อ'})
                    st.rerun()
            with st.expander('ดูรายละเอียด'):
                if (project / 'console.log').exists():
                    st.code((project / 'console.log').read_text()[-2500:])
        else:
            st.markdown(f'''<div style="
                background:#fafafa;
                border:1px solid #e5e5e5;
                border-radius:12px;
                padding:2rem;
                text-align:center;
                margin:1rem 0;
            ">
                <div style="font-size:2.5rem;margin-bottom:.75rem;">⚙</div>
                <div style="font-weight:600;font-size:1rem;color:#0a0a0a;margin-bottom:.5rem;">
                    {current.get('phase', 'กำลังประมวลผล')}
                </div>
            </div>''', unsafe_allow_html=True)
            done, total = current.get('done', 0), current.get('total', 0)
            if total:
                st.progress(min(done / total, 1.), text=f'{done:,} / {total:,}')
            st.caption('เปิดเครื่องไว้ระหว่างประมวลผล — เมื่อเสร็จหน้าจอจะเปลี่ยนอัตโนมัติ')

    progress_panel()
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — REVIEW & DOWNLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config, faces, assignments, merged, names, review = catalog(project)
work = Path(config.get('work', str(project / 'work')))
import pandas as pd
inventory = pd.read_csv(work / 'images.csv')

# ── header ──
st.title(f'{chosen}')
st.markdown(f'<p class="hero-subtitle">ตรวจสอบผลการจัดกลุ่ม แก้ไขชื่อ และส่งออกรูปเป็นโฟลเดอร์</p>', unsafe_allow_html=True)

step_indicator(3)

# ── stats ──
c1, c2, c3, c4 = st.columns(4)
c1.metric('รูปในชุด', f'{len(inventory):,}')
c2.metric('ใบหน้าที่พบ', f'{len(faces):,}')
c3.metric('กลุ่มบุคคล', f'{len(names) - 1:,}')
c4.metric('ยังไม่มีกลุ่ม', f'{int((assignments.cluster_id == -1).sum()):,}')

st.caption('จำนวนกลุ่มยังไม่ใช่จำนวนคนที่ยืนยันแล้ว — ตรวจใบหน้าและรวมกลุ่มที่เป็นคนเดียวกันได้ด้านล่าง')

# ── tabs ──
browse, export, about = st.tabs(['👤 ตรวจกลุ่มบุคคล', '📦 ดาวน์โหลดรูป', 'ℹ️ หลักการทำงาน'])

# ──────── TAB: BROWSE ────────
with browse:
    with st.expander('ลองปรับการจัดกลุ่มใหม่'):
        st.caption('ใช้ใบหน้าที่คำนวณไว้แล้ว สร้างเป็นชุดใหม่ การแก้กลุ่มในชุดเดิมยังอยู่')
        with st.form(f'recluster_{chosen}'):
            new_title = st.text_input('ชื่อชุดผลลัพธ์ใหม่', f'{chosen}_ปรับกลุ่ม')
            new_algo = st.selectbox('วิธีจัดกลุ่มใหม่', ['dbscan', 'hdbscan', 'agglomerative'])
            new_threshold = st.slider('ระยะห่างสูงสุดของใบหน้า', .05, .9, .4, .01)
            st.caption('ค่าสูงรวมกลุ่มง่ายขึ้น แต่อาจรวมคนละคน')
            if st.form_submit_button('จัดกลุ่มใหม่จากข้อมูลเดิม'):
                try:
                    with st.spinner('กำลังปรับกลุ่ม'):
                        recluster_project(project, project_path(new_title), new_algo, new_threshold, None)
                    st.session_state.pending_project = new_title
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(str(error))

    if not len(faces):
        st.info('ชุดรูปนี้ไม่มีใบหน้าที่ผ่านเกณฑ์ — ไปที่ดาวน์โหลดรูปเพื่อเก็บรูปที่ไม่พบใบหน้าได้')
    else:
        left, right = st.columns([1, 3])
        ids = list(names)
        counts = merged.groupby('cluster_id').size().to_dict()

        with left:
            st.markdown('##### บุคคล')
            query = st.text_input('ค้นหาชื่อ', key=f'query_{chosen}', placeholder='พิมพ์ชื่อ...', label_visibility='collapsed')
            visible = [i for i in ids if query.casefold() in names[i].casefold()]
            if not visible:
                st.info('ไม่มีชื่อที่ตรงกัน')
                st.stop()
            cluster = st.selectbox(
                'เลือกบุคคล', visible,
                format_func=lambda i: f'{names[i]}  ·  {counts.get(i, 0)} ใบหน้า',
                key=f'cluster_{chosen}',
                label_visibility='collapsed'
            )

            if cluster >= 0:
                st.markdown('---')
                with st.form(f'rename_{chosen}_{cluster}'):
                    new_name = st.text_input('ชื่อบุคคล', names[cluster])
                    if st.form_submit_button('บันทึกชื่อ'):
                        try:
                            edit_project(project, 'rename', cluster=cluster, name=new_name)
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

                targets = [i for i in ids if i >= 0 and i != cluster]
                if targets:
                    with st.form(f'merge_{chosen}_{cluster}'):
                        target = st.selectbox('รวมกลุ่มนี้เข้ากับ', targets, format_func=lambda i: names[i])
                        if st.form_submit_button('รวมเป็นคนเดียวกัน'):
                            edit_project(project, 'merge', source=cluster, target=target)
                            st.rerun()

            st.markdown('---')
            if st.button('↩ ย้อนการแก้ไขล่าสุด', disabled=not (project / 'review_previous.json').exists()):
                edit_project(project, 'undo')
                st.rerun()

        with right:
            group = merged[merged.cluster_id == cluster]
            st.markdown(f'##### {names[cluster]}')
            st.caption(f'{len(group):,} ใบหน้า · {group.image_path.nunique():,} รูป')

            mode = st.radio(
                'แสดง',
                ['ใบหน้าเพื่อแก้กลุ่ม', 'รูปงานของคนนี้'],
                horizontal=True,
                key=f'mode_{chosen}',
                label_visibility='collapsed'
            )

            if mode == 'ใบหน้าเพื่อแก้กลุ่ม':
                pages = max(1, (len(group) + 23) // 24)
                page = st.number_input('หน้า', 1, pages, 1, key=f'page_{chosen}_{cluster}')
                selected_faces = []
                cols = st.columns(6)
                for number, row in enumerate(group.iloc[(page - 1) * 24:page * 24].itertuples()):
                    with cols[number % 6]:
                        show_image(row.crop_path)
                        if st.checkbox(f'ใบหน้า {(page - 1) * 24 + number + 1}', key=f'face_{chosen}_{row.face_id}'):
                            selected_faces.append(row.face_id)
                        st.caption(Path(row.image_path).name)
                if selected_faces:
                    target = st.selectbox(
                        'ย้ายใบหน้าที่เลือกไป', ['new'] + ids,
                        format_func=lambda i: 'สร้างกลุ่มคนใหม่' if i == 'new' else names[i],
                        key=f'move_{chosen}_{cluster}'
                    )
                    if st.button(f'ย้าย {len(selected_faces)} ใบหน้า', type='primary'):
                        edit_project(project, 'move', faces=selected_faces, target=target)
                        st.rerun()
            else:
                photos = group.image_path.drop_duplicates().tolist()
                pages = max(1, (len(photos) + 11) // 12)
                page = st.number_input('หน้ารูป', 1, pages, 1, key=f'photo_page_{chosen}_{cluster}')
                cols = st.columns(3)
                for number, path in enumerate(photos[(page - 1) * 12:page * 12]):
                    with cols[number % 3]:
                        show_image(path, 600)
                        st.caption(Path(path).name)


# ──────── TAB: EXPORT ────────
with export:
    st.markdown('##### ดาวน์โหลดรูปที่จัดแล้ว')
    st.caption('ดาวน์โหลดเป็นไฟล์ ZIP ภายในแยกโฟลเดอร์ตามบุคคลและชื่อที่ตั้งไว้ แตก ZIP แล้วใช้รูปได้เลย')
    all_people = st.checkbox('ดาวน์โหลดทุกคน (รวมรูปที่ยังไม่ทราบบุคคลและรูปที่ไม่พบใบหน้า)', True)
    selected = None if all_people else st.multiselect(
        'เลือกบุคคล', [i for i in names if i >= 0], format_func=lambda i: names[i]
    )
    selection_key = (chosen, review['revision'], None if selected is None else tuple(sorted(selected)))
    if st.session_state.get('download_selection') != selection_key:
        st.session_state.pop('prepared_download', None)
        st.session_state.download_selection = selection_key

    try:
        _, estimate = export_project(project, ROOT/'outputs/download_preview', selected, dry_run=True)
        size = estimate['required_bytes']/2**30
        st.info(f'รูปที่จะอยู่ใน ZIP {estimate["copies"]:,} ไฟล์ · ขนาดประมาณ {size:.2f} GB')
        st.caption('รูปหลายคนจะอยู่ในโฟลเดอร์ของทุกคนที่เกี่ยวข้อง รูปต้นฉบับยังอยู่ที่เดิม')
        if st.button('เตรียมไฟล์ ZIP สำหรับดาวน์โหลด', type='primary', disabled=selected == []):
            bar = st.progress(0., text='กำลังเตรียมรูป')
            with st.spinner('กำลังรวมโฟลเดอร์เป็น ZIP...'):
                path, result = prepare_zip(project, selected, progress=lambda d,t: bar.progress(d/t, text=f'เตรียมรูป {d:,} / {t:,}'))
            bar.empty()
            st.session_state.prepared_download = {'path':str(path), 'copies':result['copies']}
        prepared = st.session_state.get('prepared_download')
        if prepared:
            @st.cache_resource
            def download_service():
                return LocalDownloads()
            zip_path = Path(prepared['path'])
            if zip_path.is_file():
                url = download_service().url(zip_path, f'{chosen}_รูปที่จัดแล้ว.zip')
                st.success(f'พร้อมดาวน์โหลด {prepared["copies"]:,} รูป')
                st.link_button('ดาวน์โหลดโฟลเดอร์รูป (.zip)', url, type='primary')
                st.caption('ไฟล์จะดาวน์โหลดผ่านเบราว์เซอร์ ไม่ต้องระบุ path ปลายทาง')
    except (ValueError, OSError) as error:
        st.error(str(error))


# ──────── TAB: ABOUT ────────
with about:
    st.markdown('##### จากรูปงาน สู่โฟลเดอร์ของแต่ละคน')

    st.markdown('''
<div style="display:flex;flex-direction:column;gap:16px;margin:1.5rem 0;">

<div style="display:flex;gap:16px;align-items:flex-start;">
    <span style="display:inline-flex;align-items:center;justify-content:center;min-width:32px;height:32px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.75rem;font-weight:700;">1</span>
    <div>
        <div style="font-weight:600;font-size:.9rem;color:#0a0a0a;">ตรวจใบหน้า</div>
        <div style="font-size:.82rem;color:#525252;margin-top:2px;">SCRFD ค้นหาตำแหน่งใบหน้าและจุดสำคัญ เช่น ตา จมูก และมุมปาก</div>
    </div>
</div>

<div style="display:flex;gap:16px;align-items:flex-start;">
    <span style="display:inline-flex;align-items:center;justify-content:center;min-width:32px;height:32px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.75rem;font-weight:700;">2</span>
    <div>
        <div style="font-weight:600;font-size:.9rem;color:#0a0a0a;">จัดแนวใบหน้า</div>
        <div style="font-size:.82rem;color:#525252;margin-top:2px;">หมุนและปรับขนาดให้ใบหน้าอยู่ในแนวเดียวกัน เพื่อเปรียบเทียบได้ง่ายขึ้น</div>
    </div>
</div>

<div style="display:flex;gap:16px;align-items:flex-start;">
    <span style="display:inline-flex;align-items:center;justify-content:center;min-width:32px;height:32px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.75rem;font-weight:700;">3</span>
    <div>
        <div style="font-weight:600;font-size:.9rem;color:#0a0a0a;">สร้างข้อมูลแทนใบหน้า</div>
        <div style="font-size:.82rem;color:#525252;margin-top:2px;">โมเดล ArcFace R50 สร้างเวกเตอร์ 512 มิติ แล้วปรับความยาวเป็น 1</div>
    </div>
</div>

<div style="display:flex;gap:16px;align-items:flex-start;">
    <span style="display:inline-flex;align-items:center;justify-content:center;min-width:32px;height:32px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.75rem;font-weight:700;">4</span>
    <div>
        <div style="font-weight:600;font-size:.9rem;color:#0a0a0a;">จัดกลุ่มใบหน้าคล้ายกัน</div>
        <div style="font-size:.82rem;color:#525252;margin-top:2px;">ใช้ DBSCAN / HDBSCAN — ใบหน้าที่คล้ายกันจะได้กลุ่มเดียวกัน ส่วนที่ไม่มั่นใจแยกไว้ตรวจ</div>
    </div>
</div>

<div style="display:flex;gap:16px;align-items:flex-start;">
    <span style="display:inline-flex;align-items:center;justify-content:center;min-width:32px;height:32px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.75rem;font-weight:700;">5</span>
    <div>
        <div style="font-weight:600;font-size:.9rem;color:#0a0a0a;">ตรวจแก้โดยผู้ใช้</div>
        <div style="font-size:.82rem;color:#525252;margin-top:2px;">เปลี่ยนชื่อ รวมกลุ่ม หรือแยกใบหน้าที่ระบบจัดผิด</div>
    </div>
</div>

<div style="display:flex;gap:16px;align-items:flex-start;">
    <span style="display:inline-flex;align-items:center;justify-content:center;min-width:32px;height:32px;border-radius:50%;background:#0a0a0a;color:#fff;font-size:.75rem;font-weight:700;">6</span>
    <div>
        <div style="font-weight:600;font-size:.9rem;color:#0a0a0a;">ส่งออกรูป</div>
        <div style="font-size:.82rem;color:#525252;margin-top:2px;">คัดลอกรูปต้นฉบับเข้าทุกกลุ่มบุคคลที่ปรากฏในรูป โดยไม่ซ้ำในโฟลเดอร์เดียวกัน</div>
    </div>
</div>

</div>
''', unsafe_allow_html=True)

    st.caption('รุ่นนี้ใช้โมเดลสำเร็จรูปทั้งตัวตรวจและตัวเปรียบเทียบใบหน้า · ชื่อบุคคลต้องกำหนดโดยผู้ใช้')
    st.caption('ผลอาจคลาดเคลื่อนเมื่อใบหน้าเล็ก เบลอ หันข้าง หรือมีคนหน้าคล้ายกัน')
