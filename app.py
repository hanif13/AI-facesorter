"""Thai local UI for finding, reviewing and exporting event photos by person."""
from __future__ import annotations
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st
from src.common.image_io import image_paths, load_rgb
from src.product import (ROOT, PROJECTS, DEFAULT_SOURCE, read_json, write_json, project_path,
                         import_sample, catalog, edit_project, export_project, recluster_project)

st.set_page_config(page_title='Face Sorter • จัดรูปตามคน', page_icon='📷', layout='wide')
st.markdown('''<style>
.block-container {max-width:1240px; padding-top:2rem;}
h1 {letter-spacing:-.035em;} [data-testid="stMetric"] {background:#eef6f5;border-radius:14px;padding:16px;}
[data-testid="stSidebar"] {background:#f2f5f7;}
</style>''', unsafe_allow_html=True)


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


sample = import_sample()
PROJECTS.mkdir(parents=True, exist_ok=True)
projects = sorted(p for p in PROJECTS.iterdir() if p.is_dir() and ((p/'project.json').exists() or (p/'status.json').exists()))
with st.sidebar:
    st.title('📷 Face Sorter')
    st.caption('รูปงานของคุณ ค้นตามคนได้ง่ายขึ้น')
    choices = ['สร้างชุดรูปใหม่'] + [p.name for p in projects]
    pending = st.session_state.pop('pending_project', None)
    if pending in choices:
        st.session_state.project_choice = pending
    default = 'day3_ready' if 'day3_ready' in choices else ('day3_product' if 'day3_product' in choices else (sample.name if sample else choices[0]))
    if st.session_state.get('project_choice') not in choices:
        st.session_state.project_choice = default
    chosen = st.selectbox('ชุดรูป', choices, key='project_choice')
    st.divider()
    st.caption('1 · เลือกโฟลเดอร์รูป\n\n2 · ตรวจและแก้กลุ่มบุคคล\n\n3 · ส่งออกรูปที่ต้องการ')
    st.caption('ทำงานในเครื่อง • ไม่อัปโหลดรูปไปบริการภายนอก')

st.title('จัดรูปงาน ตามคนที่อยู่ในภาพ')
st.caption('ค้นหาใบหน้า จัดกลุ่ม และตรวจแก้ก่อนส่งออก รูปหลายคนจะอยู่ได้ในหลายโฟลเดอร์')

if chosen == 'สร้างชุดรูปใหม่':
    st.subheader('เริ่มจากโฟลเดอร์รูปของคุณ')
    with st.form('new_project'):
        source_text = st.text_input('ตำแหน่งโฟลเดอร์รูป', DEFAULT_SOURCE)
        name = st.text_input('ชื่อชุดรูป', 'งานใหม่_' + datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%m%d_%H%M'))
        with st.expander('ตัวเลือกการจัดกลุ่ม'):
            alg = st.selectbox('วิธีจัดกลุ่ม', ['dbscan', 'hdbscan', 'agglomerative'])
            threshold = st.slider('ระยะห่างสูงสุด (DBSCAN / Agglomerative)', .05, .9, .4, .01)
            rescue = st.checkbox('ช่วยนำใบหน้าที่ยังไม่มีกลุ่มเข้ากลุ่มที่คล้ายกัน', False)
            similarity = st.slider('ความคล้ายขั้นต่ำสำหรับนำเข้ากลุ่ม', .2, .9, .45, .01)
            st.caption('ค่าต่ำอาจรวมคนละคน ค่าสูงอาจแยกคนเดียวกันเป็นหลายกลุ่ม กรุณาตรวจผล')
        submitted = st.form_submit_button('ค้นหาและจัดกลุ่มใบหน้า', type='primary')
    if submitted:
        try:
            source = Path(source_text).expanduser().resolve()
            if not source.is_dir() or not image_paths(source):
                raise ValueError('ไม่พบโฟลเดอร์รูป หรือโฟลเดอร์ไม่มีรูปที่รองรับ')
            project = project_path(name)
            if project.exists():
                raise ValueError('ชื่อชุดรูปนี้มีอยู่แล้ว กรุณาเปิดจากเมนูหรือใช้ชื่อใหม่')
            project.mkdir(parents=True)
            write_json(project / 'status.json', {'state':'running','phase':'กำลังเริ่มงาน','done':0,'total':0})
            command = [sys.executable, str(ROOT/'run_pipeline.py'), '--input', str(source), '--run-name', name,
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

project = project_path(chosen)
status = read_json(project/'status.json', {'state':'running','phase':'กำลังเริ่มงาน'})
if status.get('state') != 'ready':
    @st.fragment(run_every='3s')
    def progress_panel():
        current = read_json(project/'status.json', status)
        proc = st.session_state.get(f'job_{chosen}')
        if proc and proc.poll() is not None and current.get('state') == 'running':
            current = {'state':'failed','error':'งานหยุดก่อนเสร็จ กรุณาลองทำต่อ'}
        if current.get('state') == 'ready':
            st.rerun()
        elif current.get('state') == 'failed':
            st.error(current.get('error', 'ประมวลผลไม่สำเร็จ'))
            if st.button('ลองทำต่อจากขั้นตอนที่เสร็จแล้ว'):
                config = read_json(project/'project.json')
                if config and config.get('source'):
                    command = [sys.executable, str(ROOT/'run_pipeline.py'), '--input', config['source'], '--run-name', chosen,
                               '--algo', config.get('algorithm','hdbscan'), '--threshold', str(config.get('threshold',.45))]
                    if config.get('rescue_sim',.45) is None:
                        command.append('--no-rescue')
                    else:
                        command += ['--rescue-sim',str(config.get('rescue_sim',.45))]
                    if config.get('sample_limit'):
                        command += ['--limit',str(config['sample_limit'])]
                    with (project/'console.log').open('a') as log:
                        st.session_state[f'job_{chosen}'] = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write_json(project/'status.json', {'state':'running','phase':'กำลังทำต่อ'})
                    st.rerun()
            with st.expander('รายละเอียดสำหรับแก้ปัญหา'):
                if (project/'console.log').exists():
                    st.code((project/'console.log').read_text()[-2500:])
        else:
            st.info(current.get('phase','กำลังประมวลผล'))
            done, total = current.get('done',0), current.get('total',0)
            if total:
                st.progress(min(done/total,1.), text=f'{done:,} / {total:,}')
            st.caption('เปิดเครื่องไว้ระหว่างประมวลผล เมื่อเสร็จหน้าจอจะเปลี่ยนเป็นกลุ่มบุคคล')
    progress_panel()
    st.stop()

config, faces, assignments, merged, names, review = catalog(project)
work = Path(config.get('work', str(project/'work')))
import pandas as pd
inventory = pd.read_csv(work/'images.csv')
a,b,c,d = st.columns(4)
a.metric('รูปในชุด', f'{len(inventory):,}')
b.metric('ใบหน้าที่พบ', f'{len(faces):,}')
c.metric('กลุ่มบุคคล', f'{len(names)-1:,}')
d.metric('ใบหน้ายังไม่มีกลุ่ม', f'{int((assignments.cluster_id==-1).sum()):,}')
st.caption('จำนวนกลุ่มยังไม่ใช่จำนวนคนที่ยืนยันแล้ว ตรวจใบหน้าและรวมกลุ่มที่เป็นคนเดียวกันได้ด้านล่าง')

browse, export, about = st.tabs(['ตรวจกลุ่มบุคคล', 'ส่งออกรูป', 'หลักการทำงาน'])
with browse:
    with st.expander('ลองปรับการจัดกลุ่มใหม่'):
        st.caption('ใช้ใบหน้าที่คำนวณไว้แล้ว สร้างเป็นชุดใหม่ การแก้กลุ่มในชุดเดิมยังอยู่')
        with st.form(f'recluster_{chosen}'):
            new_title = st.text_input('ชื่อชุดผลลัพธ์ใหม่', f'{chosen}_ปรับกลุ่ม')
            new_algo = st.selectbox('วิธีจัดกลุ่มใหม่', ['dbscan','hdbscan','agglomerative'])
            new_threshold = st.slider('ระยะห่างสูงสุดของใบหน้า',.05,.9,.4,.01)
            st.caption('ค่าสูงรวมกลุ่มง่ายขึ้น แต่อาจรวมคนละคน โปรดตรวจใบหน้าหลังปรับ')
            if st.form_submit_button('จัดกลุ่มใหม่จากข้อมูลเดิม'):
                try:
                    with st.spinner('กำลังปรับกลุ่ม'):
                        recluster_project(project,project_path(new_title),new_algo,new_threshold,None)
                    st.session_state.pending_project=new_title
                    st.rerun()
                except (ValueError,OSError) as error:
                    st.error(str(error))
    if not len(faces):
        st.info('ชุดรูปนี้ไม่มีใบหน้าที่ผ่านเกณฑ์ ไปที่ส่งออกรูปเพื่อเก็บรูปที่ไม่พบใบหน้าได้')
    else:
        left, right = st.columns([1,3])
        ids = list(names)
        counts = merged.groupby('cluster_id').size().to_dict()
        with left:
            query = st.text_input('ค้นหาชื่อกลุ่ม', key=f'query_{chosen}')
            visible = [i for i in ids if query.casefold() in names[i].casefold()]
            if not visible:
                st.info('ไม่มีชื่อที่ตรงกัน')
                st.stop()
            cluster = st.selectbox('เลือกบุคคล', visible,
                format_func=lambda i:f'{names[i]} · {counts.get(i,0)} ใบหน้า', key=f'cluster_{chosen}')
            if cluster >= 0:
                with st.form(f'rename_{chosen}_{cluster}'):
                    new_name = st.text_input('ชื่อบุคคล', names[cluster])
                    if st.form_submit_button('บันทึกชื่อ'):
                        try:
                            edit_project(project,'rename',cluster=cluster,name=new_name); st.rerun()
                        except ValueError as error:
                            st.error(str(error))
                targets = [i for i in ids if i>=0 and i!=cluster]
                if targets:
                    with st.form(f'merge_{chosen}_{cluster}'):
                        target = st.selectbox('รวมกลุ่มนี้เข้ากับ', targets,format_func=lambda i:names[i])
                        if st.form_submit_button('รวมเป็นคนเดียวกัน'):
                            edit_project(project,'merge',source=cluster,target=target); st.rerun()
            if st.button('ย้อนการแก้ไขล่าสุด', disabled=not (project/'review_previous.json').exists()):
                edit_project(project,'undo'); st.rerun()
        with right:
            group = merged[merged.cluster_id == cluster]
            st.subheader(names[cluster])
            st.caption(f'{len(group):,} ใบหน้า ใน {group.image_path.nunique():,} รูป')
            mode = st.radio('แสดง', ['ใบหน้าเพื่อแก้กลุ่ม','รูปงานของคนนี้'],horizontal=True,key=f'mode_{chosen}')
            if mode == 'ใบหน้าเพื่อแก้กลุ่ม':
                pages = max(1,(len(group)+23)//24)
                page = st.number_input('หน้า',1,pages,1,key=f'page_{chosen}_{cluster}')
                selected_faces = []
                cols = st.columns(6)
                for number,row in enumerate(group.iloc[(page-1)*24:page*24].itertuples()):
                    with cols[number%6]:
                        show_image(row.crop_path)
                        if st.checkbox(f'ใบหน้า {(page-1)*24+number+1}',key=f'face_{chosen}_{row.face_id}'):
                            selected_faces.append(row.face_id)
                        st.caption(Path(row.image_path).name)
                if selected_faces:
                    target = st.selectbox('ย้ายใบหน้าที่เลือกไป', ['new']+ids,
                        format_func=lambda i:'สร้างกลุ่มคนใหม่' if i=='new' else names[i],key=f'move_{chosen}_{cluster}')
                    if st.button(f'ย้าย {len(selected_faces)} ใบหน้า',type='primary'):
                        edit_project(project,'move',faces=selected_faces,target=target); st.rerun()
            else:
                photos = group.image_path.drop_duplicates().tolist()
                pages = max(1,(len(photos)+11)//12)
                page = st.number_input('หน้ารูป',1,pages,1,key=f'photo_page_{chosen}_{cluster}')
                cols = st.columns(3)
                for number,path in enumerate(photos[(page-1)*12:page*12]):
                    with cols[number%3]:
                        show_image(path,600); st.caption(Path(path).name)

with export:
    st.subheader('คัดลอกรูปเป็นโฟลเดอร์ตามบุคคล')
    st.write('ชื่อที่ตั้งและการแก้กลุ่มจะใช้กับการส่งออกครั้งใหม่ รูปต้นฉบับจะยังอยู่ที่เดิม')
    all_people = st.checkbox('ส่งออกทุกคน รวมรูปที่ยังไม่ทราบบุคคลและรูปที่ไม่พบใบหน้า',True)
    selected = None if all_people else st.multiselect('เลือกบุคคล',[i for i in names if i>=0],format_func=lambda i:names[i])
    target_text = st.text_input('โฟลเดอร์ปลายทาง',str(ROOT/'outputs/sorted_photos'/f'{chosen}_review{review["revision"]}'))
    try:
        _, estimate = export_project(project,Path(target_text),selected,dry_run=True)
        st.info(f'จะคัดลอก {estimate["copies"]:,} ไฟล์ ใช้พื้นที่ประมาณ {estimate["required_bytes"]/2**30:.2f} GB · พื้นที่ว่าง {estimate["free_bytes"]/2**30:.2f} GB')
        if st.button('ส่งออกรูป',type='primary',disabled=selected==[]):
            with st.spinner('กำลังคัดลอกรูป'):
                _, result = export_project(project,Path(target_text),selected)
            st.success(f'ส่งออกแล้ว {result["copies"]:,} ไฟล์ ที่ {Path(target_text).expanduser().resolve()}')
    except (ValueError,OSError) as error:
        st.error(str(error))

with about:
    st.subheader('จากรูปงาน สู่โฟลเดอร์ของแต่ละคน')
    st.markdown('''1. **ตรวจใบหน้า** — SCRFD ค้นหาตำแหน่งใบหน้าและจุดสำคัญ เช่น ตา จมูก และมุมปาก
2. **จัดแนวใบหน้า** — หมุนและปรับขนาดให้ใบหน้าอยู่ในแนวเดียวกัน เพื่อเปรียบเทียบได้ง่ายขึ้น
3. **สร้างข้อมูลแทนใบหน้า** — โมเดล ArcFace R50 ผ่าน InsightFace สร้างเวกเตอร์ 512 มิติ แล้วปรับความยาวเป็น 1
4. **จัดกลุ่มใบหน้าคล้ายกัน** — ใช้ HDBSCAN หรือวิธีที่เลือก ใบหน้าที่คล้ายกันจะได้กลุ่มเดียวกัน ส่วนที่ไม่มั่นใจแยกไว้ตรวจ
5. **ตรวจแก้โดยผู้ใช้** — เปลี่ยนชื่อ รวมกลุ่ม หรือแยกใบหน้าที่ระบบจัดผิด
6. **ส่งออกรูป** — คัดลอกรูปต้นฉบับเข้าทุกกลุ่มบุคคลที่ปรากฏในรูป โดยไม่ซ้ำในโฟลเดอร์เดียวกัน''')
    st.caption('รุ่นนี้ใช้โมเดลสำเร็จรูปทั้งตัวตรวจและตัวเปรียบเทียบใบหน้า ไม่มีการฝึกโมเดลเพิ่ม ชื่อบุคคลต้องกำหนดโดยผู้ใช้')
    st.caption('ผลอาจคลาดเคลื่อนเมื่อใบหน้าเล็ก เบลอ หันข้าง หรือมีคนหน้าคล้ายกัน ยังไม่มีผลวัดความแม่นยำกับคำตอบอ้างอิงของรูปงาน')
