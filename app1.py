import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import random
import string
import os
import base64
import re
import numpy as np
from pathlib import Path
from io import BytesIO

# --- Supabase imports ---
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# --- sklearn for AI (optional) ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    class TfidfVectorizer:
        def fit_transform(self, texts):
            return np.zeros((len(texts), 1))
    def cosine_similarity(a, b):
        return np.zeros((a.shape[0], b.shape[0]))

st.set_page_config(page_title="Student Evaluation System", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# ---------- Supabase configuration ----------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://qjlypajeavbmsobhogfd.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "sb_publishable_yfvTRBgiBGYHWgrX2ZbzVw_KLTk0ddh")
USE_SUPABASE = SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY

if USE_SUPABASE:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.sidebar.success("✅ Connected to Supabase – your data is now persistent!")
else:
    st.sidebar.error("⚠️ Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY in secrets.")
    st.stop()

# ---------- Session state ----------
if 'current_student' not in st.session_state:
    st.session_state.current_student = None
if 'current_teacher' not in st.session_state:
    st.session_state.current_teacher = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'reset_email' not in st.session_state:
    st.session_state.reset_email = None
if 'temp_password' not in st.session_state:
    st.session_state.temp_password = None
if 'view_file' not in st.session_state:
    st.session_state.view_file = None
if 'submission_review' not in st.session_state:
    st.session_state.submission_review = None
if 'page' not in st.session_state:
    st.session_state.page = "Welcome"
if 'show_privacy' not in st.session_state:
    st.session_state.show_privacy = False
if 'show_terms' not in st.session_state:
    st.session_state.show_terms = False
if 'show_contact' not in st.session_state:
    st.session_state.show_contact = False
if 'show_deletion' not in st.session_state:
    st.session_state.show_deletion = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'edit_submission_id' not in st.session_state:
    st.session_state.edit_submission_id = None
if 'edit_activity_id' not in st.session_state:
    st.session_state.edit_activity_id = None
if 'view_submission' not in st.session_state:
    st.session_state.view_submission = None
if 'view_activity' not in st.session_state:
    st.session_state.view_activity = None

# ---------- Helper functions ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def generate_temp_password(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def validate_class_name(class_name):
    if not class_name or len(class_name.strip()) < 2:
        return False, "Class name is too short. Use e.g., BCA VI"
    class_upper = class_name.upper()
    patterns = [
        r'^(BA|BCom|BSC|BCA|MCA|MA|MCom|MSC|BBA|MBA|B TECH|M TECH|DIPLOMA)\s*([IVXLCDM]+|\d+)$',
        r'^(SEMESTER|SEM)\s*([IVXLCDM]+|\d+)$',
        r'^(\d+)(?:ST|ND|RD|TH)?\s*YEAR$',
        r'^CLASS\s*(\d+)$'
    ]
    for pattern in patterns:
        match = re.match(pattern, class_upper)
        if match:
            parts = match.groups()
            if len(parts) == 2:
                prefix, suffix = parts
                roman_map = {'I':'I','II':'II','III':'III','IV':'IV','V':'V','VI':'VI',
                             'VII':'VII','VIII':'VIII','IX':'IX','X':'X'}
                if suffix in roman_map:
                    normalized = f"{prefix} {roman_map[suffix]}"
                elif suffix.isdigit():
                    num = int(suffix)
                    if 1 <= num <= 10:
                        roman_numerals = ['I','II','III','IV','V','VI','VII','VIII','IX','X']
                        normalized = f"{prefix} {roman_numerals[num-1]}"
                    else:
                        normalized = f"{prefix} {suffix}"
                else:
                    normalized = f"{prefix} {suffix}"
                return True, ' '.join(normalized.split())
    if len(class_name) <= 30 and re.match(r'^[A-Za-z0-9\s\-]+$', class_name):
        return True, class_name.upper()
    return False, "Invalid class name format. Use e.g., BCA VI"

# ---------- Supabase CRUD functions ----------
# (All the existing CRUD functions remain unchanged. For brevity, they are not repeated here, but they must be included in the final code.)
# Please copy the full set of functions from the previous version.

# ... (Insert all the previous CRUD, AI, and helper functions here) ...

# ---------- New function to export to Excel ----------
def export_to_excel(df, sheet_name="Data"):
    """Convert dataframe to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# ---------- Create uploads directory ----------
Path("uploads").mkdir(exist_ok=True)

# ========== STREAMLIT UI ==========
# (The entire Streamlit UI from the previous version, with the teacher's View Submissions section modified to include Excel download buttons.)
# ... (Keep all the existing UI code exactly as before, except modify the teacher's "📂 View Submissions" section as shown below.)

# ========== TEACHER SECTION ==========
# Only the modified part is shown here. The rest of the teacher section remains unchanged.
elif st.session_state.page == "📂 View Submissions":
    st.header("📂 Student Work for Evaluation - View, Edit & Delete")
    tab1, tab2 = st.tabs(["📝 Assignments (Submissions)", "🎯 Extra Activities"])

    # ---------- TAB 1: ASSIGNMENTS ----------
    with tab1:
        subs_df = get_all_submissions_for_teacher()
        if not subs_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                class_filter = st.selectbox("Filter by Class", ["All"] + subs_df['class'].unique().tolist(), key="assign_class")
            with col2:
                subject_filter = st.selectbox("Filter by Subject", ["All"] + subs_df['subject'].unique().tolist(), key="assign_subj")
            with col3:
                student_filter = st.selectbox("Filter by Student", ["All"] + subs_df['student_name'].unique().tolist(), key="assign_student")
            
            filtered = subs_df.copy()
            if class_filter != "All":
                filtered = filtered[filtered['class'] == class_filter]
            if subject_filter != "All":
                filtered = filtered[filtered['subject'] == subject_filter]
            if student_filter != "All":
                filtered = filtered[filtered['student_name'] == student_filter]
            
            st.write(f"**Total Assignments:** {len(filtered)}")
            
            # Excel download button
            if not filtered.empty:
                # Prepare data for export (select columns suitable for Excel)
                export_df = filtered[['student_name', 'reg_no', 'class', 'subject', 'title', 'submission_type', 'date', 'points_earned', 'grade', 'ai_confidence', 'plagiarism_score']].copy()
                export_df.columns = ['Student Name', 'Reg No', 'Class', 'Subject', 'Title', 'Type', 'Date', 'Points', 'Grade', 'AI Confidence', 'Plagiarism Score']
                # Convert confidence to percentage if needed
                export_df['AI Confidence'] = (export_df['AI Confidence'] * 100).round(0).astype(str) + '%'
                export_df['Plagiarism Score'] = (export_df['Plagiarism Score'] * 100).round(0).astype(str) + '%'
                
                excel_data = export_to_excel(export_df, "Assignments")
                st.download_button(
                    label="📥 Download Assignments as Excel",
                    data=excel_data,
                    file_name=f"assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Display table with actions
            for idx, row in filtered.iterrows():
                with st.container():
                    cols = st.columns([1.2, 1.2, 1.2, 1.5, 2, 1, 1, 1, 0.8, 0.8, 0.8])
                    cols[0].write(row['student_name'])
                    cols[1].write(row['reg_no'])
                    cols[2].write(row['class'])
                    cols[3].write(row['subject'])
                    cols[4].write(row['title'][:40] + "..." if len(row['title']) > 40 else row['title'])
                    cols[5].write(row['date'])
                    cols[6].write(str(row['points_earned']))
                    cols[7].write(row['grade'] if row.get('grade') else "N/A")
                    
                    if cols[8].button("👁️", key=f"view_sub_{row['submission_id']}", help="View Details"):
                        st.session_state.view_submission = row.to_dict()
                        st.rerun()
                    
                    if cols[9].button("✏️", key=f"edit_sub_{row['submission_id']}", help="Edit"):
                        st.session_state.edit_submission_id = row['submission_id']
                        st.session_state.edit_submission_data = row.to_dict()
                        st.rerun()
                    
                    if cols[10].button("🗑️", key=f"del_sub_{row['submission_id']}", help="Delete Submission"):
                        if delete_submission(row['submission_id'], row['file_path']):
                            st.success(f"Deleted submission: {row['title']}")
                            st.rerun()
                        else:
                            st.error("Failed to delete submission.")
                    
                    st.markdown("---")
            
            # View/Edit modals (unchanged)
            # ... (keep the same modal code as before)
        else:
            st.info("No assignment submissions found.")

    # ---------- TAB 2: EXTRA ACTIVITIES ----------
    with tab2:
        acts_df = get_all_activities_for_teacher()
        if not acts_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                class_filter_act = st.selectbox("Filter by Class", ["All"] + acts_df['class'].unique().tolist(), key="act_class")
            with col2:
                type_filter = st.selectbox("Filter by Activity Type", ["All"] + acts_df['activity_type'].unique().tolist(), key="act_type")
            with col3:
                student_filter_act = st.selectbox("Filter by Student", ["All"] + acts_df['student_name'].unique().tolist(), key="act_student")
            
            filtered_act = acts_df.copy()
            if class_filter_act != "All":
                filtered_act = filtered_act[filtered_act['class'] == class_filter_act]
            if type_filter != "All":
                filtered_act = filtered_act[filtered_act['activity_type'] == type_filter]
            if student_filter_act != "All":
                filtered_act = filtered_act[filtered_act['student_name'] == student_filter_act]
            
            st.write(f"**Total Extra Activities:** {len(filtered_act)}")
            
            # Excel download button
            if not filtered_act.empty:
                export_df_act = filtered_act[['student_name', 'reg_no', 'class', 'activity_type', 'topic', 'date', 'duration_minutes', 'points_earned', 'remarks']].copy()
                export_df_act.columns = ['Student Name', 'Reg No', 'Class', 'Activity Type', 'Topic', 'Date', 'Duration (min)', 'Points', 'Remarks']
                excel_data_act = export_to_excel(export_df_act, "Extra Activities")
                st.download_button(
                    label="📥 Download Extra Activities as Excel",
                    data=excel_data_act,
                    file_name=f"extra_activities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Display table with actions
            for idx, row in filtered_act.iterrows():
                with st.container():
                    cols = st.columns([1.2, 1.2, 1.2, 1.5, 2, 1, 1, 0.8, 0.8, 0.8])
                    cols[0].write(row['student_name'])
                    cols[1].write(row['reg_no'])
                    cols[2].write(row['class'])
                    cols[3].write(row['activity_type'])
                    cols[4].write(row['topic'][:40] + "..." if len(row['topic']) > 40 else row['topic'])
                    cols[5].write(row['date'])
                    cols[6].write(str(row['points_earned']))
                    
                    if cols[7].button("👁️", key=f"view_act_{row['activity_id']}", help="View Details"):
                        st.session_state.view_activity = row.to_dict()
                        st.rerun()
                    
                    if cols[8].button("✏️", key=f"edit_act_{row['activity_id']}", help="Edit"):
                        st.session_state.edit_activity_id = row['activity_id']
                        st.session_state.edit_activity_data = row.to_dict()
                        st.rerun()
                    
                    if cols[9].button("🗑️", key=f"del_act_{row['activity_id']}", help="Delete Activity"):
                        if delete_activity(row['activity_id'], row['file_path']):
                            st.success(f"Deleted activity: {row['topic']}")
                            st.rerun()
                        else:
                            st.error("Failed to delete activity.")
                    
                    st.markdown("---")
            
            # View/Edit modals (unchanged)
            # ... (keep the same modal code as before)
        else:
            st.info("No extra activities submitted yet.")

# ========== FOOTER TABS ==========
# (Keep the footer as before)
