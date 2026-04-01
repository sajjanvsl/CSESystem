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
def get_student_by_email_or_regno(identifier, use_regno=False):
    try:
        if use_regno:
            result = supabase.table('students').select('*').eq('reg_no', identifier.strip()).execute()
        else:
            result = supabase.table('students').select('*').eq('email', identifier.strip().lower()).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        st.error(f"DB error: {e}")
        return None

def add_student(reg_no, name, class_name, email, password, phone=None):
    is_valid, normalized_class = validate_class_name(class_name)
    if not is_valid:
        st.error(normalized_class)
        return False
    try:
        existing = supabase.table('students').select('student_id').eq('reg_no', reg_no).execute()
        if existing.data:
            st.error("Registration number already exists!")
            return False
        existing = supabase.table('students').select('student_id').eq('email', email).execute()
        if existing.data:
            st.error("Email already exists!")
            return False
        data = {
            'reg_no': reg_no,
            'name': name,
            'class': normalized_class,
            'email': email,
            'phone': phone,
            'password': hash_password(password),
            'last_active': datetime.now().strftime('%Y-%m-%d')
        }
        result = supabase.table('students').insert(data).execute()
        if result.data:
            st.success(f"Registration successful! Class set to {normalized_class}")
            return True
        else:
            st.error("Registration failed.")
            return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def edit_student_registration(student_id, name, class_name, email, phone):
    is_valid, normalized_class = validate_class_name(class_name)
    if not is_valid:
        st.error(normalized_class)
        return False
    try:
        existing = supabase.table('students').select('student_id').eq('email', email).neq('student_id', student_id).execute()
        if existing.data:
            st.error("Email already exists for another student!")
            return False
        result = supabase.table('students').update({
            'name': name,
            'class': normalized_class,
            'email': email,
            'phone': phone
        }).eq('student_id', student_id).execute()
        if result.data:
            st.success(f"Registration updated. Class set to {normalized_class}")
            st.session_state.current_student = result.data[0]
            return True
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def faculty_edit_student(student_id, reg_no, name, class_name, email, phone, password=None):
    is_valid, normalized_class = validate_class_name(class_name)
    if not is_valid:
        st.error(normalized_class)
        return False
    try:
        existing = supabase.table('students').select('student_id').eq('reg_no', reg_no).neq('student_id', student_id).execute()
        if existing.data:
            st.error("Registration number already exists for another student!")
            return False
        existing = supabase.table('students').select('student_id').eq('email', email).neq('student_id', student_id).execute()
        if existing.data:
            st.error("Email already exists for another student!")
            return False
        update_data = {
            'reg_no': reg_no,
            'name': name,
            'class': normalized_class,
            'email': email,
            'phone': phone
        }
        if password:
            update_data['password'] = hash_password(password)
        result = supabase.table('students').update(update_data).eq('student_id', student_id).execute()
        if result.data:
            st.success("Student updated successfully.")
            return True
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def delete_student_complete(student_id, student_reg_no):
    """Complete deletion of student and all associated data with file cleanup."""
    try:
        # Get all submissions with file paths
        submissions = supabase.table('submissions').select('file_path').eq('student_id', student_id).execute()
        for sub in submissions.data:
            if sub.get('file_path') and os.path.exists(sub['file_path']):
                os.remove(sub['file_path'])
        
        # Get all activities with file paths
        activities = supabase.table('activities').select('file_path').eq('student_id', student_id).execute()
        for act in activities.data:
            if act.get('file_path') and os.path.exists(act['file_path']):
                os.remove(act['file_path'])
        
        # Delete from all related tables
        supabase.table('submissions').delete().eq('student_id', student_id).execute()
        supabase.table('activities').delete().eq('student_id', student_id).execute()
        supabase.table('daily_activity').delete().eq('student_id', student_id).execute()
        supabase.table('rewards').delete().eq('student_id', student_id).execute()
        supabase.table('point_transactions').delete().eq('student_id', student_id).execute()
        supabase.table('student_subjects').delete().eq('student_id', student_id).execute()
        
        # Delete student folder from uploads
        student_folder = Path("uploads") / student_reg_no
        if student_folder.exists():
            import shutil
            shutil.rmtree(student_folder)
        
        # Finally delete student record
        supabase.table('students').delete().eq('student_id', student_id).execute()
        
        return True
    except Exception as e:
        st.error(f"Error deleting student: {e}")
        return False

def get_all_students():
    try:
        result = supabase.table('students').select('*').order('total_points', desc=True).execute()
        if result.data:
            return pd.DataFrame(result.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_student_subjects(student_id):
    try:
        result = supabase.table('student_subjects').select('subject_id, registration_date, subjects(*)').eq('student_id', student_id).eq('status', 'Active').execute()
        if not result.data:
            return pd.DataFrame()
        rows = []
        for item in result.data:
            subj = item['subjects']
            rows.append({
                'subject_id': subj['subject_id'],
                'subject_code': subj['subject_code'],
                'subject_name': subj['subject_name'],
                'class': subj['class'],
                'teacher_id': subj['teacher_id'],
                'teacher_name': None,
                'registration_date': item['registration_date']
            })
        teacher_ids = [row['teacher_id'] for row in rows if row['teacher_id']]
        if teacher_ids:
            teachers = supabase.table('teachers').select('teacher_id, name').in_('teacher_id', teacher_ids).execute()
            teacher_map = {t['teacher_id']: t['name'] for t in teachers.data}
            for row in rows:
                if row['teacher_id']:
                    row['teacher_name'] = teacher_map.get(row['teacher_id'])
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame()

def register_student_subjects(student_id, subject_ids):
    try:
        for sid in subject_ids:
            supabase.table('student_subjects').upsert({
                'student_id': student_id,
                'subject_id': sid,
                'status': 'Active'
            }, on_conflict='student_id,subject_id').execute()
        return True
    except Exception as e:
        st.error(f"Error registering subjects: {e}")
        return False

def remove_student_subject(student_id, subject_id):
    try:
        supabase.table('student_subjects').delete().eq('student_id', student_id).eq('subject_id', subject_id).execute()
        return True
    except Exception as e:
        st.error(f"Error removing subject: {e}")
        return False

def add_teacher(teacher_code, name, email, password, department):
    try:
        existing = supabase.table('teachers').select('teacher_id').eq('teacher_code', teacher_code).execute()
        if existing.data:
            st.error("Teacher code already exists!")
            return False
        existing = supabase.table('teachers').select('teacher_id').eq('email', email).execute()
        if existing.data:
            st.error("Email already exists!")
            return False
        data = {
            'teacher_code': teacher_code,
            'name': name,
            'email': email,
            'password': hash_password(password),
            'department': department
        }
        result = supabase.table('teachers').insert(data).execute()
        if result.data:
            st.success("Teacher registered successfully!")
            return True
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def authenticate_teacher(email, password):
    try:
        result = supabase.table('teachers').select('*').eq('email', email.strip().lower()).execute()
        if not result.data:
            return None
        teacher = result.data[0]
        if teacher['password'] == hash_password(password):
            return teacher
        return None
    except Exception as e:
        return None

def get_all_teachers():
    try:
        result = supabase.table('teachers').select('teacher_id, teacher_code, name, email, department').order('name').execute()
        if result.data:
            return pd.DataFrame(result.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def add_subject(subject_code, subject_name, class_name, teacher_id=None):
    is_valid, normalized_class = validate_class_name(class_name)
    if not is_valid:
        st.error(normalized_class)
        return False
    try:
        existing = supabase.table('subjects').select('subject_id').eq('subject_code', subject_code).execute()
        if existing.data:
            st.error("Subject code already exists!")
            return False
        data = {
            'subject_code': subject_code,
            'subject_name': subject_name,
            'class': normalized_class,
            'teacher_id': teacher_id
        }
        result = supabase.table('subjects').insert(data).execute()
        if result.data:
            st.success(f"Subject created for class {normalized_class}")
            return True
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def delete_subject(subject_id):
    try:
        result = supabase.table('student_subjects').select('id').eq('subject_id', subject_id).execute()
        if result.data:
            if not st.session_state.get(f'confirm_delete_{subject_id}', False):
                st.session_state[f'confirm_delete_{subject_id}'] = True
                st.warning(f"⚠️ This subject has {len(result.data)} student registrations. Delete anyway?")
                return False
        supabase.table('student_subjects').delete().eq('subject_id', subject_id).execute()
        supabase.table('subjects').delete().eq('subject_id', subject_id).execute()
        st.success("Subject deleted.")
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def get_all_subjects(class_name=None):
    try:
        if class_name:
            is_valid, norm = validate_class_name(class_name)
            candidates = [class_name, norm] if is_valid else [class_name]
            rows = []
            for cls in candidates:
                res = supabase.table('subjects').select('*, teachers(name)').eq('class', cls).execute()
                if res.data:
                    rows = res.data
                    break
            if not rows:
                res = supabase.table('subjects').select('*, teachers(name)').ilike('class', f'%{class_name}%').execute()
                rows = res.data
        else:
            res = supabase.table('subjects').select('*, teachers(name)').order('class').order('subject_name').execute()
            rows = res.data
        df = pd.DataFrame(rows)
        if not df.empty and 'teachers' in df.columns:
            df['teacher_name'] = df['teachers'].apply(lambda x: x['name'] if x else None)
        return df
    except Exception as e:
        return pd.DataFrame()

def assign_subject_to_teacher(subject_id, teacher_id):
    try:
        supabase.table('subjects').update({'teacher_id': teacher_id}).eq('subject_id', subject_id).execute()
        return True
    except Exception as e:
        return False

def add_submission(submission_data):
    try:
        result = supabase.table('submissions').insert(submission_data).execute()
        return result.data[0]['submission_id'] if result.data else None
    except Exception as e:
        st.error(f"Error saving submission: {e}")
        return None

def update_submission(submission_id, title, description, points_earned=None, grade=None):
    try:
        update_data = {'title': title, 'description': description}
        if points_earned is not None:
            update_data['points_earned'] = points_earned
        if grade is not None:
            update_data['grade'] = grade
        result = supabase.table('submissions').update(update_data).eq('submission_id', submission_id).execute()
        return result.data is not None
    except Exception as e:
        st.error(f"Error updating submission: {e}")
        return False

def update_activity(activity_id, topic, remarks, points_earned=None):
    try:
        update_data = {'topic': topic, 'remarks': remarks}
        if points_earned is not None:
            update_data['points_earned'] = points_earned
        result = supabase.table('activities').update(update_data).eq('activity_id', activity_id).execute()
        return result.data is not None
    except Exception as e:
        st.error(f"Error updating activity: {e}")
        return False

def delete_submission(submission_id, file_path=None):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        supabase.table('submissions').delete().eq('submission_id', submission_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting submission: {e}")
        return False

def delete_activity(activity_id, file_path=None):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        supabase.table('activities').delete().eq('activity_id', activity_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting activity: {e}")
        return False

def get_student_submissions(student_id):
    try:
        result = supabase.table('submissions').select('*').eq('student_id', student_id).order('date', desc=True).execute()
        if result.data:
            return pd.DataFrame(result.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_all_submissions_for_teacher():
    try:
        result = supabase.table('submissions').select('*, students(name, reg_no, class)').order('date', desc=True).execute()
        if not result.data:
            return pd.DataFrame()
        rows = []
        for s in result.data:
            rows.append({
                'submission_id': s['submission_id'],
                'student_id': s['student_id'],
                'submission_type': s['submission_type'],
                'subject': s['subject'],
                'title': s['title'],
                'description': s['description'],
                'date': s['date'],
                'file_path': s['file_path'],
                'file_name': s['file_name'],
                'file_type': s['file_type'],
                'file_size': s['file_size'],
                'ai_confidence': s['ai_confidence'],
                'ai_feedback': s['ai_feedback'],
                'plagiarism_score': s['plagiarism_score'],
                'points_earned': s['points_earned'],
                'grade': s['grade'],
                'student_name': s['students']['name'],
                'reg_no': s['students']['reg_no'],
                'class': s['students']['class']
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame()

def get_all_activities_for_teacher():
    try:
        result = supabase.table('activities').select(
            '*, students(name, reg_no, class)'
        ).order('date', desc=True).execute()
        if not result.data:
            return pd.DataFrame()
        rows = []
        for act in result.data:
            rows.append({
                'activity_id': act['activity_id'],
                'student_id': act['student_id'],
                'activity_type': act['activity_type'],
                'topic': act['topic'],
                'date': act['date'],
                'duration_minutes': act['duration_minutes'],
                'remarks': act['remarks'],
                'points_earned': act['points_earned'],
                'file_path': act['file_path'],
                'file_name': act['file_name'],
                'student_name': act['students']['name'],
                'reg_no': act['students']['reg_no'],
                'class': act['students']['class']
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching activities: {e}")
        return pd.DataFrame()

def add_activity(activity_data):
    try:
        supabase.table('activities').insert(activity_data).execute()
        return True
    except Exception as e:
        st.error(f"Error adding activity: {e}")
        return False

def get_student_activities(student_id):
    try:
        result = supabase.table('activities').select('*').eq('student_id', student_id).order('date', desc=True).execute()
        if result.data:
            return pd.DataFrame(result.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def add_reward_claim(student_id, reward_type, points_cost):
    try:
        supabase.table('rewards').insert({
            'student_id': student_id,
            'reward_type': reward_type,
            'points_cost': points_cost,
            'reward_date': datetime.now().strftime('%Y-%m-%d'),
            'status': 'Claimed',
            'claimed_at': datetime.now().isoformat()
        }).execute()
        student = supabase.table('students').select('total_points').eq('student_id', student_id).execute().data[0]
        supabase.table('students').update({'total_points': student['total_points'] - points_cost}).eq('student_id', student_id).execute()
        supabase.table('point_transactions').insert({
            'student_id': student_id,
            'transaction_type': 'Reward Claimed',
            'points': -points_cost,
            'description': f"Claimed {reward_type}"
        }).execute()
        return True
    except Exception as e:
        return False

def get_reward_history(student_id):
    try:
        result = supabase.table('rewards').select('*').eq('student_id', student_id).order('reward_date', desc=True).execute()
        if result.data:
            return pd.DataFrame(result.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_daily_activity(student_id, days=7):
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        result = supabase.table('daily_activity').select('*').eq('student_id', student_id).gte('activity_date', cutoff).order('activity_date', desc=True).execute()
        if result.data:
            return pd.DataFrame(result.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def update_daily_activity(student_id, date, points, activity_type='submission'):
    try:
        existing = supabase.table('daily_activity').select('*').eq('student_id', student_id).eq('activity_date', date).execute()
        if existing.data:
            row = existing.data[0]
            if activity_type == 'submission':
                supabase.table('daily_activity').update({
                    'submission_count': row['submission_count'] + 1,
                    'total_points_earned': row['total_points_earned'] + points
                }).eq('log_id', row['log_id']).execute()
            else:
                supabase.table('daily_activity').update({
                    'activity_count': row['activity_count'] + 1,
                    'total_points_earned': row['total_points_earned'] + points
                }).eq('log_id', row['log_id']).execute()
        else:
            new_row = {
                'student_id': student_id,
                'activity_date': date,
                'submission_count': 1 if activity_type == 'submission' else 0,
                'activity_count': 1 if activity_type != 'submission' else 0,
                'total_points_earned': points
            }
            supabase.table('daily_activity').insert(new_row).execute()
    except Exception as e:
        pass

def get_leaderboard(limit=20, class_filter=None):
    try:
        query = supabase.table('students').select('student_id, reg_no, name, class, total_points, current_streak, best_streak')
        if class_filter and class_filter != "All Classes":
            query = query.eq('class', class_filter)
        result = query.order('total_points', desc=True).order('current_streak', desc=True).limit(limit).execute()
        if not result.data:
            return pd.DataFrame()
        df = pd.DataFrame(result.data)
        for idx, row in df.iterrows():
            subs = supabase.table('submissions').select('submission_id').eq('student_id', row['student_id']).execute()
            acts = supabase.table('activities').select('activity_id').eq('student_id', row['student_id']).execute()
            df.loc[idx, 'submissions_total'] = len(subs.data)
            df.loc[idx, 'activities_count'] = len(acts.data)
        df.insert(0, 'Rank', range(1, len(df)+1))
        return df
    except Exception as e:
        return pd.DataFrame()

def get_student_progress(student_id):
    try:
        subs = supabase.table('submissions').select('points_earned').eq('student_id', student_id).execute()
        total_submissions = len(subs.data)
        submission_points = sum(s.get('points_earned', 0) for s in subs.data)
        acts = supabase.table('activities').select('points_earned').eq('student_id', student_id).execute()
        total_activities = len(acts.data)
        activity_points = sum(a.get('points_earned', 0) for a in acts.data)
        return {
            'total_submissions': total_submissions,
            'submission_points': submission_points,
            'total_activities': total_activities,
            'activity_points': activity_points
        }
    except Exception as e:
        return {'total_submissions':0, 'submission_points':0, 'total_activities':0, 'activity_points':0}

def update_student_streak(student_id, submission_date):
    try:
        student = supabase.table('students').select('last_active, current_streak, best_streak').eq('student_id', student_id).execute()
        if not student.data:
            return
        last_active = student.data[0]['last_active']
        current_streak = student.data[0]['current_streak']
        best_streak = student.data[0]['best_streak']
        if last_active == submission_date:
            new_streak = current_streak + 1
            new_best = max(best_streak, new_streak)
            supabase.table('students').update({
                'current_streak': new_streak,
                'best_streak': new_best,
                'last_active': submission_date
            }).eq('student_id', student_id).execute()
        else:
            supabase.table('students').update({
                'current_streak': 1,
                'last_active': submission_date
            }).eq('student_id', student_id).execute()
    except Exception as e:
        pass

# ---------- AI & file functions ----------
def validate_submission_with_ai(submission_text, subject, topic=None):
    if not submission_text or len(submission_text.strip()) < 10:
        return {
            'confidence': 0.3,
            'feedback': "Submission is too short. Please provide more detailed content.",
            'plagiarism_score': 0.0,
            'quality_score': 0.3,
            'word_count': len(submission_text.split()),
            'keyword_score': 0.0
        }
    word_count = len(submission_text.split())
    sentence_count = len(re.findall(r'[.!?]+', submission_text))
    try:
        if topic:
            refs = supabase.table('reference_answers').select('answer_text').eq('subject', subject).eq('topic', topic).execute()
        else:
            refs = supabase.table('reference_answers').select('answer_text').eq('subject', subject).execute()
        references = [r['answer_text'] for r in refs.data] if refs.data else []
    except Exception:
        references = []
    similarity_scores = []
    if references and SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            all_texts = [submission_text] + references
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            similarity_scores = similarity_matrix.flatten().tolist()
        except:
            similarity_scores = []
    elif references and not SKLEARN_AVAILABLE:
        submission_words = set(submission_text.lower().split())
        for ref in references:
            ref_words = set(ref.lower().split())
            if len(submission_words) > 0 and len(ref_words) > 0:
                overlap = len(submission_words.intersection(ref_words))
                total = len(submission_words.union(ref_words))
                similarity_scores.append(overlap / total if total > 0 else 0)
            else:
                similarity_scores.append(0)
    plagiarism_score = max(similarity_scores) if similarity_scores else 0.0
    common_keywords = {
        'Database Management': ['sql','query','table','database','normalization','index','data','server'],
        'Web Technologies': ['html','css','javascript','web','browser','server','client','http'],
        'Python Programming': ['python','variable','function','class','loop','list','dict','import'],
        'General': ['example','explain','define','describe','compare','analyze','discuss']
    }
    keywords = common_keywords.get(subject, common_keywords['General'])
    submission_lower = submission_text.lower()
    keyword_matches = sum(1 for keyword in keywords if keyword in submission_lower)
    keyword_score = keyword_matches / len(keywords) if keywords else 0.5
    length_score = min(word_count / 100, 1.0)
    structure_score = min(sentence_count / 5, 1.0)
    quality_score = (length_score * 0.3 + structure_score * 0.2 + keyword_score * 0.5)
    confidence = quality_score * 0.7 + (1 - plagiarism_score) * 0.3
    feedback_parts = []
    if word_count < 50:
        feedback_parts.append("• Your submission could be more detailed. Aim for at least 50-100 words.")
    elif word_count > 200:
        feedback_parts.append("• Good length! Your submission is comprehensive.")
    if plagiarism_score > 0.7:
        feedback_parts.append("⚠️ High similarity with reference materials detected. Please use your own words.")
    elif plagiarism_score > 0.4:
        feedback_parts.append("• Moderate similarity with reference materials. Try to paraphrase more.")
    else:
        feedback_parts.append("✓ Good originality in your response.")
    if keyword_score < 0.3:
        feedback_parts.append("• Missing key terminology. Try to include more subject-specific terms.")
    elif keyword_score > 0.7:
        feedback_parts.append("✓ Excellent use of subject terminology!")
    if structure_score < 0.5:
        feedback_parts.append("• Consider organizing your response into clearer sentences/paragraphs.")
    if not SKLEARN_AVAILABLE:
        feedback_parts.append("• Note: Advanced AI features limited (scikit-learn not installed).")
    feedback = "\n".join(feedback_parts)
    return {
        'confidence': round(confidence, 2),
        'feedback': feedback,
        'plagiarism_score': round(plagiarism_score, 2),
        'quality_score': round(quality_score, 2),
        'word_count': word_count,
        'keyword_score': round(keyword_score, 2)
    }

def add_reference_answer(subject, topic, answer_text, teacher_id):
    try:
        supabase.table('reference_answers').insert({
            'subject': subject,
            'topic': topic,
            'answer_text': answer_text,
            'created_by': teacher_id
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error adding reference answer: {e}")
        return False

def check_duplicate_submission(student_id, subject, title, description, submission_type):
    try:
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        res = supabase.table('submissions').select('submission_id, date, title').eq('student_id', student_id).eq('subject', subject).eq('title', title).gte('date', thirty_days_ago).execute()
        if res.data:
            return True, f"You have already submitted an assignment with the same title on {res.data[0]['date']}"
        return False, ""
    except Exception as e:
        return False, ""

def get_auto_grade_points(submission_type):
    mapping = {
        'Daily Homework': 5,
        'Weekly Assignment': 15,
        'Monthly Assignment': 30,
        'Seminar': 10,
        'Project': 15,
        'Research Paper': 25,
        'Lab Report': 8,
        'Extra Activity': 25
    }
    return mapping.get(submission_type, 5)

def get_auto_grade_letter(submission_type):
    mapping = {
        'Daily Homework': 'A',
        'Weekly Assignment': 'A',
        'Monthly Assignment': 'A+',
        'Seminar': 'A',
        'Project': 'A+',
        'Research Paper': 'A+',
        'Lab Report': 'A',
        'Extra Activity': 'A+'
    }
    return mapping.get(submission_type, 'A')

def add_submission_with_ai(student_id, submission_type, subject, title, description, date,
                           file_path=None, file_name=None, file_type=None, file_size=None):
    points = get_auto_grade_points(submission_type)
    grade = get_auto_grade_letter(submission_type)
    is_duplicate, dup_msg = check_duplicate_submission(student_id, subject, title, description, submission_type)
    if is_duplicate:
        st.error(dup_msg)
        return None
    ai_result = validate_submission_with_ai(description, subject)
    adjusted_points = points * ai_result['confidence']
    data = {
        'student_id': student_id,
        'submission_type': submission_type,
        'subject': subject,
        'title': title,
        'description': description,
        'date': date,
        'file_path': file_path,
        'file_name': file_name,
        'file_type': file_type,
        'file_size': file_size,
        'max_points': points,
        'points_earned': adjusted_points,
        'grade': grade,
        'status': 'Graded',
        'auto_graded': 1,
        'graded_at': datetime.now().isoformat(),
        'ai_confidence': ai_result['confidence'],
        'ai_feedback': ai_result['feedback'],
        'plagiarism_score': ai_result['plagiarism_score']
    }
    submission_id = add_submission(data)
    if submission_id:
        student = supabase.table('students').select('total_points').eq('student_id', student_id).execute().data[0]
        supabase.table('students').update({'total_points': student['total_points'] + adjusted_points}).eq('student_id', student_id).execute()
        update_student_streak(student_id, date)
        supabase.table('point_transactions').insert({
            'student_id': student_id,
            'transaction_type': 'Auto Graded',
            'points': adjusted_points,
            'description': f"AI-graded: {submission_type}",
            'reference_id': submission_id
        }).execute()
        update_daily_activity(student_id, date, adjusted_points, 'submission')
        st.session_state.submission_review = ai_result
        return submission_id
    return None

def add_extra_activity(student_id, activity_type, topic, date, duration, remarks,
                       file_path=None, file_name=None):
    points = 25
    data = {
        'student_id': student_id,
        'activity_type': 'Extra Activity',
        'topic': topic,
        'date': date,
        'duration_minutes': duration,
        'remarks': remarks,
        'points_earned': points,
        'file_path': file_path,
        'file_name': file_name
    }
    if add_activity(data):
        student = supabase.table('students').select('total_points').eq('student_id', student_id).execute().data[0]
        supabase.table('students').update({'total_points': student['total_points'] + points}).eq('student_id', student_id).execute()
        supabase.table('point_transactions').insert({
            'student_id': student_id,
            'transaction_type': 'Extra Activity',
            'points': points,
            'description': f"Extra Activity: {topic}"
        }).execute()
        update_daily_activity(student_id, date, points, 'activity')
        return True
    return False

def forgot_password(email, user_type):
    email = email.strip().lower()
    try:
        if user_type == 'student':
            res = supabase.table('students').select('student_id, name').eq('email', email).execute()
        else:
            res = supabase.table('teachers').select('teacher_id, name').eq('email', email).execute()
        if not res.data:
            return False, "Email not found"
        temp_pass = generate_temp_password()
        hashed = hash_password(temp_pass)
        if user_type == 'student':
            supabase.table('students').update({'password': hashed}).eq('email', email).execute()
        else:
            supabase.table('teachers').update({'password': hashed}).eq('email', email).execute()
        expires = (datetime.now() + timedelta(hours=24)).isoformat()
        supabase.table('password_reset').insert({
            'email': email,
            'reset_code': temp_pass,
            'expires_at': expires
        }).execute()
        return True, temp_pass
    except Exception as e:
        return False, str(e)

def reset_password(email, new_password):
    email = email.strip().lower()
    try:
        res = supabase.table('students').select('student_id').eq('email', email).execute()
        if res.data:
            supabase.table('students').update({'password': hash_password(new_password)}).eq('email', email).execute()
            return True, 'student'
        res = supabase.table('teachers').select('teacher_id').eq('email', email).execute()
        if res.data:
            supabase.table('teachers').update({'password': hash_password(new_password)}).eq('email', email).execute()
            return True, 'teacher'
        return False, None
    except Exception as e:
        return False, None

def request_data_deletion(email, user_type, reason):
    try:
        supabase.table('deletion_requests').insert({
            'email': email,
            'user_type': user_type,
            'reason': reason,
            'status': 'Pending'
        }).execute()
        return True
    except Exception as e:
        return False

def get_file_download_link(file_path, file_name):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            bytes_data = f.read()
            b64 = base64.b64encode(bytes_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
            return href
    return None

def get_file_view_link(file_path, file_name, file_type):
    if os.path.exists(file_path):
        if file_type and file_type.startswith('image/'):
            with open(file_path, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                return f'<img src="data:{file_type};base64,{b64}" style="max-width:100%; max-height:300px;">'
        elif file_type == 'application/pdf':
            with open(file_path, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                return f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="500px"></iframe>'
        elif file_type and file_type.startswith('text/'):
            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return f'<pre style="background:#f5f5f5; padding:10px;">{content}</pre>'
    return None

# ---------- Create uploads directory ----------
Path("uploads").mkdir(exist_ok=True)

# ---------- Streamlit UI ----------
st.title("📚 Continuous Student Evaluation & Monitoring System")
st.markdown("---")

# Sidebar
with st.sidebar:
    if st.session_state.user_role:
        if st.session_state.user_role == "student":
            student = st.session_state.current_student
            if student:
                st.header("🎓 Student Info")
                st.success(f"**{student['name']}**")
                st.info(f"Reg No: {student['reg_no']}")
                st.info(f"Class: {student['class']}")
                st.info(f"Email: {student['email']}")
                st.info(f"Points: {student['total_points']} 🏆")
                st.info(f"Streak: {student['current_streak']} days 🔥")
                if st.button("✏️ Edit Registration"):
                    st.session_state.page = "edit_registration"
                    st.rerun()
            if st.button("Logout"):
                st.session_state.current_student = None
                st.session_state.current_teacher = None
                st.session_state.user_role = None
                st.session_state.logged_in = False
                st.session_state.page = "Welcome"
                st.rerun()
        elif st.session_state.user_role == "teacher":
            teacher = st.session_state.current_teacher
            if teacher:
                st.header("👨‍🏫 Teacher Info")
                st.success(f"**Prof. {teacher['name']}**")
                st.info(f"Email: {teacher['email']}")
                st.info(f"Dept: {teacher['department']}")
            if st.button("Logout"):
                st.session_state.current_student = None
                st.session_state.current_teacher = None
                st.session_state.user_role = None
                st.session_state.logged_in = False
                st.session_state.page = "Welcome"
                st.rerun()
    else:
        st.header("🔐 Login")
        login_tab1, login_tab2, login_tab3 = st.tabs(["Student Login", "Teacher Login", "Forgot Password"])

        with login_tab1:
            with st.expander("Test Credentials (use these)"):
                st.write("**Student:** test@student.com / test123")
                st.write("**Registration No:** TEST001 / test123")
            login_method = st.radio("Login with:", ["Email", "Registration Number"])
            if login_method == "Email":
                email = st.text_input("Email", key="student_email")
                password = st.text_input("Password", type="password", key="student_pass")
                if st.button("Student Login"):
                    if email and password:
                        student = get_student_by_email_or_regno(email, use_regno=False)
                        if student and verify_password(password, student['password']):
                            st.session_state.current_student = student
                            st.session_state.user_role = "student"
                            st.session_state.logged_in = True
                            st.session_state.page = "🏠 Dashboard"
                            st.rerun()
                        else:
                            st.error("Invalid email or password!")
            else:
                reg_no = st.text_input("Registration Number", key="student_regno")
                password = st.text_input("Password", type="password", key="student_pass_reg")
                if st.button("Student Login"):
                    if reg_no and password:
                        student = get_student_by_email_or_regno(reg_no, use_regno=True)
                        if student and verify_password(password, student['password']):
                            st.session_state.current_student = student
                            st.session_state.user_role = "student"
                            st.session_state.logged_in = True
                            st.session_state.page = "🏠 Dashboard"
                            st.rerun()
                        else:
                            st.error("Invalid registration number or password!")

        with login_tab2:
            with st.expander("Test Credentials"):
                st.write("**Teacher:** test@teacher.com / test123")
            email = st.text_input("Email", key="teacher_email")
            password = st.text_input("Password", type="password", key="teacher_pass")
            if st.button("Teacher Login"):
                if email and password:
                    teacher = authenticate_teacher(email, password)
                    if teacher:
                        st.session_state.current_teacher = teacher
                        st.session_state.user_role = "teacher"
                        st.session_state.logged_in = True
                        st.session_state.page = "🏠 Teacher Dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid email or password!")

        with login_tab3:
            st.subheader("🔑 Forgot Password")
            with st.form("forgot_password_form"):
                fp_email = st.text_input("Enter your registered email", key="fp_email").strip()
                fp_user_type = st.selectbox("I am a", ["Student", "Teacher"], key="fp_type")
                submitted = st.form_submit_button("Reset Password")
                if submitted:
                    if not fp_email:
                        st.error("Please enter your email address.")
                    else:
                        success, result = forgot_password(fp_email, fp_user_type.lower())
                        if success:
                            st.success("✅ Password reset successfully!")
                            st.info(f"🔑 Your temporary password is: `{result}`")
                            st.warning("Please copy this password and log in. Change it after logging in.")
                        else:
                            st.error(f"❌ {result}")

        with st.expander("🔧 Debug"):
            if st.button("🚀 Direct Login as Test Student"):
                student = get_student_by_email_or_regno("test@student.com")
                if student:
                    st.session_state.current_student = student
                    st.session_state.user_role = "student"
                    st.session_state.logged_in = True
                    st.session_state.page = "🏠 Dashboard"
                    st.rerun()
                else:
                    st.error("Test student not found.")

    st.markdown("---")
    if st.session_state.user_role:
        st.header("📱 Navigation")
        if st.session_state.user_role == "student":
            if st.session_state.page != "edit_registration":
                selected = st.radio("Go to:", [
                    "🏠 Dashboard", "📚 My Subjects", "➕ New Submission", "➕ Extra Activity",
                    "📋 My Submissions", "📂 My Uploads", "📈 Daily Activity", "🏆 Leaderboard",
                    "🎁 Rewards", "👤 Edit Profile"
                ])
                if selected != st.session_state.page:
                    st.session_state.page = selected
                    st.rerun()
            else:
                st.radio("Go to:", [
                    "🏠 Dashboard", "📚 My Subjects", "➕ New Submission", "➕ Extra Activity",
                    "📋 My Submissions", "📂 My Uploads", "📈 Daily Activity", "🏆 Leaderboard",
                    "🎁 Rewards", "👤 Edit Profile"
                ], index=0, disabled=True)
                if st.button("← Back to Dashboard"):
                    st.session_state.page = "🏠 Dashboard"
                    st.rerun()
        else:
            selected = st.radio("Go to:", [
                "🏠 Teacher Dashboard", "📚 Subject Management", "👨‍🎓 Manage Students",
                "📂 View Submissions", "📊 Class Analytics", "🏆 Leaderboard", "👤 Edit Profile",
                "⚙️ Manage System", "🤖 AI Reference Answers"
            ])
            if selected != st.session_state.page:
                st.session_state.page = selected
                st.rerun()
    else:
        if st.session_state.page != "Welcome":
            st.session_state.page = "Welcome"

# ========== WELCOME PAGE ==========
if st.session_state.page == "Welcome":
    st.header("Welcome to Student Evaluation System")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Student Registration")
        st.write("- Register with email and password")
        st.write("- Choose your subjects (dynamic classes)")
        st.write("- Submit assignments and earn points")
        st.write("- AI-powered validation system")
        st.write("- Duplicate submission prevention")
        st.write("- Track your progress")
        st.info("Class name examples: BCA VI, BA II, BCom I, MCA III")
        with st.expander("New Student Registration"):
            with st.form("new_student_form"):
                reg_no = st.text_input("Registration Number*")
                name = st.text_input("Full Name*")
                class_name = st.text_input("Class*", placeholder="e.g., BCA VI, BA II, BCom I")
                email = st.text_input("Email*")
                phone = st.text_input("Phone")
                password = st.text_input("Password*", type="password")
                confirm = st.text_input("Confirm Password*", type="password")
                if st.form_submit_button("Register"):
                    if reg_no and name and class_name and email and password:
                        if password == confirm:
                            if add_student(reg_no, name, class_name, email, password, phone):
                                st.success("Registration successful! Please login.")
                            else:
                                st.error("Registration failed.")
                        else:
                            st.error("Passwords do not match!")
                    else:
                        st.error("Please fill all required fields.")
    with col2:
        st.subheader("👨‍🏫 Teacher Registration")
        st.write("- Register with email and password")
        st.write("- Create and manage subjects")
        st.write("- Assign subjects to yourself")
        st.write("- Monitor student performance")
        st.write("- Manage AI reference answers")
        with st.expander("Teacher Registration"):
            with st.form("teacher_reg_form"):
                t_code = st.text_input("Teacher Code*")
                t_name = st.text_input("Full Name*")
                t_email = st.text_input("Email*")
                t_password = st.text_input("Password*", type="password")
                t_confirm = st.text_input("Confirm Password*", type="password")
                t_dept = st.text_input("Department*", placeholder="e.g., History, Mathematics")
                if st.form_submit_button("Register as Teacher"):
                    if all([t_code, t_name, t_email, t_password, t_confirm, t_dept]):
                        if t_password == t_confirm:
                            if add_teacher(t_code, t_name, t_email, t_password, t_dept):
                                st.success("Registration successful! Please login.")
                            else:
                                st.error("Teacher code or email already exists!")
                        else:
                            st.error("Passwords do not match!")
                    else:
                        st.error("Please fill all fields.")

# ========== STUDENT SECTION ==========
# (Keep the entire student section from your previous code)
# For brevity, I'm not repeating it here, but it must be included in the final file.
# The student section is unchanged from the version that worked.

# ========== TEACHER SECTION ==========
elif st.session_state.user_role == "teacher":
    teacher = st.session_state.current_teacher
    if not teacher:
        st.error("Please login first!")
        st.stop()

    teacher_id = teacher['teacher_id']
    teacher_name = teacher['name']
    teacher_email = teacher['email']
    teacher_dept = teacher['department']

    if st.session_state.page == "🏠 Teacher Dashboard":
        st.header(f"Teacher Dashboard 👨‍🏫")
        total_students = supabase.table('students').select('student_id').execute()
        total_submissions = supabase.table('submissions').select('submission_id').execute()
        total_points = supabase.table('students').select('total_points').execute()
        total_points_sum = sum(s['total_points'] for s in total_points.data) if total_points.data else 0
        my_subjects = supabase.table('subjects').select('subject_id').eq('teacher_id', teacher_id).execute()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(total_students.data))
        col2.metric("Total Submissions", len(total_submissions.data))
        col3.metric("Total Points Awarded", total_points_sum)
        col4.metric("My Subjects", len(my_subjects.data))
        st.markdown("---")
        st.subheader("📚 My Subjects")
        subjects_df = get_all_subjects()
        my_subjects_df = subjects_df[subjects_df['teacher_id'] == teacher_id] if not subjects_df.empty else pd.DataFrame()
        if not my_subjects_df.empty:
            st.dataframe(my_subjects_df[['subject_code','subject_name','class']], use_container_width=True)
        else:
            st.info("You haven't been assigned any subjects yet.")
        st.subheader("📊 Class Distribution")
        class_dist = supabase.table('students').select('class').execute()
        if class_dist.data:
            df = pd.DataFrame(class_dist.data)
            dist = df['class'].value_counts().reset_index()
            dist.columns = ['Class', 'Count']
            st.dataframe(dist, use_container_width=True)

    elif st.session_state.page == "📚 Subject Management":
        # ... (unchanged, same as previous)
        st.header("📚 Subject Management")
        st.info("Class name format: e.g., BCA VI, BA II")
        tab1, tab2, tab3 = st.tabs(["➕ Create Subject", "📋 My Subjects", "👥 Assign Teachers"])
        with tab1:
            with st.form("create_subject_form"):
                col1, col2 = st.columns(2)
                with col1:
                    subject_code = st.text_input("Subject Code*", placeholder="e.g., MATH101")
                    subject_name = st.text_input("Subject Name*", placeholder="e.g., Calculus")
                with col2:
                    class_name = st.text_input("Class*", placeholder="e.g., BCA VI")
                    assign_to_self = st.checkbox("Assign this subject to me")
                if st.form_submit_button("Create Subject"):
                    if subject_code and subject_name and class_name:
                        teacher_id_to_assign = teacher_id if assign_to_self else None
                        add_subject(subject_code, subject_name, class_name, teacher_id_to_assign)
                    else:
                        st.error("Please fill all required fields.")
        with tab2:
            st.subheader("Subjects I Teach")
            subjects_df = get_all_subjects()
            my_subjects = subjects_df[subjects_df['teacher_id'] == teacher_id] if not subjects_df.empty else pd.DataFrame()
            if not my_subjects.empty:
                st.dataframe(my_subjects[['subject_code','subject_name','class','created_at']], use_container_width=True)
                st.markdown("---")
                st.subheader("🗑️ Delete Subjects")
                st.warning("Deleting a subject will also remove all student registrations.")
                subject_options = {f"{row['subject_code']} - {row['subject_name']} ({row['class']})": row['subject_id'] for _, row in my_subjects.iterrows()}
                selected = st.selectbox("Select subject to delete:", list(subject_options.keys()))
                if selected:
                    subj_id = subject_options[selected]
                    regs = supabase.table('student_subjects').select('id').eq('subject_id', subj_id).execute()
                    if regs.data:
                        st.warning(f"⚠️ This subject has {len(regs.data)} student registrations.")
                    if st.button("🗑️ Delete Subject", type="secondary"):
                        if delete_subject(subj_id):
                            st.rerun()
            else:
                st.info("You haven't been assigned any subjects yet.")
        with tab3:
            st.subheader("Assign Subjects to Teachers")
            teachers_df = get_all_teachers()
            subjects_df = get_all_subjects()
            unassigned = subjects_df[pd.isna(subjects_df['teacher_id'])] if not subjects_df.empty else pd.DataFrame()
            if not teachers_df.empty and not unassigned.empty:
                col1, col2 = st.columns(2)
                with col1:
                    selected_teacher = st.selectbox("Select Teacher", teachers_df['teacher_id'].tolist(), format_func=lambda x: teachers_df[teachers_df['teacher_id']==x]['name'].iloc[0])
                with col2:
                    selected_subject = st.selectbox("Select Subject", unassigned['subject_id'].tolist(), format_func=lambda x: f"{unassigned[unassigned['subject_id']==x]['subject_code'].iloc[0]} - {unassigned[unassigned['subject_id']==x]['subject_name'].iloc[0]}")
                if st.button("Assign Subject"):
                    if assign_subject_to_teacher(selected_subject, selected_teacher):
                        st.success("Subject assigned successfully!")
                        st.rerun()
            else:
                if teachers_df.empty: st.info("No teachers available.")
                if unassigned.empty: st.info("No unassigned subjects.")

    elif st.session_state.page == "👨‍🎓 Manage Students":
        st.header("Manage Students")
        tab1, tab2 = st.tabs(["📝 Edit Student Details", "🗑️ Delete Student Accounts"])
        with tab1:
            # ... (unchanged, same as previous)
            st.info("Faculty: You can edit all student details.")
            students_df = get_all_students()
            if not students_df.empty:
                st.subheader("All Students")
                st.dataframe(students_df[['reg_no','name','class','email','phone','total_points']], use_container_width=True)
                st.markdown("---")
                st.subheader("Edit Student Details (Faculty)")
                col1, col2 = st.columns(2)
                with col1:
                    selected_reg = st.selectbox("Select Student by Registration Number", students_df['reg_no'].tolist())
                    if selected_reg:
                        student_data = students_df[students_df['reg_no']==selected_reg].iloc[0]
                        student_id = student_data['student_id']
                        with st.form("faculty_edit_student_form"):
                            reg_no = st.text_input("Registration Number", value=student_data['reg_no'])
                            name = st.text_input("Name", value=student_data['name'])
                            class_name = st.text_input("Class", value=student_data['class'])
                            email = st.text_input("Email", value=student_data['email'])
                            phone = st.text_input("Phone", value=student_data['phone'])
                            st.subheader("Reset Password (Optional)")
                            new_password = st.text_input("New Password", type="password", help="Leave blank to keep current")
                            if st.form_submit_button("💾 Update Student"):
                                if reg_no and name and class_name and email:
                                    if faculty_edit_student(student_id, reg_no, name, class_name, email, phone, new_password if new_password else None):
                                        st.rerun()
                                else:
                                    st.error("Please fill all required fields.")
                with col2:
                    if selected_reg:
                        student = supabase.table('students').select('*').eq('reg_no', selected_reg).execute().data[0]
                        st.subheader("Student Details")
                        st.info(f"**Reg No:** {student['reg_no']}\n**Name:** {student['name']}\n**Class:** {student['class']}\n**Email:** {student['email']}\n**Phone:** {student['phone']}\n**Total Points:** {student['total_points']}\n**Current Streak:** {student['current_streak']} days")
                        with st.expander("View Registered Subjects"):
                            subj = get_student_subjects(student['student_id'])
                            if not subj.empty:
                                st.dataframe(subj[['subject_code','subject_name','teacher_name']])
                            else:
                                st.info("No subjects registered.")
                        with st.expander("View Student Submissions"):
                            subs = get_student_submissions(student['student_id'])
                            if not subs.empty:
                                st.dataframe(subs[['submission_type','subject','title','date','grade','points_earned']])
                            else:
                                st.info("No submissions yet.")
            else:
                st.info("No students found.")
        
        with tab2:
            st.warning("⚠️ **Warning: Student Deletion** - This action is permanent and cannot be undone!")
            st.info("Deleting a student will remove:\n- All submissions and uploaded files\n- All activities\n- All points and rewards\n- All subject registrations\n- The student account itself")
            students_df = get_all_students()
            if not students_df.empty:
                st.subheader("Select Student to Delete")
                # Show duplicate registrations warning
                duplicate_check = students_df.groupby(['name', 'class']).size().reset_index(name='count')
                duplicates = duplicate_check[duplicate_check['count'] > 1]
                if not duplicates.empty:
                    st.error("⚠️ **Duplicate Registrations Detected!**")
                    st.write("The following students have multiple registrations:")
                    for _, dup in duplicates.iterrows():
                        st.write(f"- **{dup['name']}** in **{dup['class']}** (Registered {dup['count']} times)")
                col1, col2 = st.columns([2, 1])
                with col1:
                    student_options = []
                    for _, row in students_df.iterrows():
                        student_options.append(f"{row['reg_no']} - {row['name']} ({row['class']})")
                    selected_student = st.selectbox("Select Student to Delete", student_options)
                    if selected_student:
                        reg_no = selected_student.split(" - ")[0]
                        student_data = students_df[students_df['reg_no'] == reg_no].iloc[0]
                        st.subheader("Student Information")
                        st.write(f"**Name:** {student_data['name']}")
                        st.write(f"**Registration No:** {student_data['reg_no']}")
                        st.write(f"**Class:** {student_data['class']}")
                        st.write(f"**Email:** {student_data['email']}")
                        st.write(f"**Total Points:** {student_data['total_points']}")
                        st.write(f"**Current Streak:** {student_data['current_streak']} days")
                        submissions = supabase.table('submissions').select('submission_id').eq('student_id', student_data['student_id']).execute()
                        activities = supabase.table('activities').select('activity_id').eq('student_id', student_data['student_id']).execute()
                        st.write(f"**Submissions:** {len(submissions.data)}")
                        st.write(f"**Activities:** {len(activities.data)}")
                with col2:
                    st.subheader("Confirm Deletion")
                    confirm_name = st.text_input("Type student name to confirm:", placeholder="Enter full name")
                    confirm_reg = st.text_input("Type registration number to confirm:", placeholder="Enter reg number")
                    
                    if selected_student:
                        reg_no = selected_student.split(" - ")[0]
                        student_data = students_df[students_df['reg_no'] == reg_no].iloc[0]
                        # Strip and compare case-insensitively for name, exact for reg number
                        if confirm_name.strip().lower() == student_data['name'].lower() and confirm_reg.strip() == student_data['reg_no']:
                            if st.button("🗑️ **PERMANENTLY DELETE STUDENT**", type="primary", use_container_width=True):
                                if delete_student_complete(student_data['student_id'], student_data['reg_no']):
                                    st.success(f"✅ Student {student_data['name']} (Reg: {student_data['reg_no']}) has been permanently deleted!")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error("Failed to delete student. Please try again.")
                        elif confirm_name or confirm_reg:
                            st.error("❌ Name or registration number does not match. Please enter correctly.")
                        else:
                            st.info("🔒 Type the student's name and registration number exactly as shown to enable deletion.")
                st.markdown("---")
                st.caption("💡 Tip: Check for duplicate registrations above. Students with multiple entries can be safely deleted, keeping only one active account.")
            else:
                st.info("No students found in the system.")

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
                
                # Create a clean table with all actions in the same row
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
                        
                        # View button
                        if cols[8].button("👁️", key=f"view_sub_{row['submission_id']}", help="View Details"):
                            st.session_state.view_submission = row.to_dict()
                            st.rerun()
                        
                        # Edit button
                        if cols[9].button("✏️", key=f"edit_sub_{row['submission_id']}", help="Edit"):
                            st.session_state.edit_submission_id = row['submission_id']
                            st.session_state.edit_submission_data = row.to_dict()
                            st.rerun()
                        
                        # Delete button
                        if cols[10].button("🗑️", key=f"del_sub_{row['submission_id']}", help="Delete Submission"):
                            if delete_submission(row['submission_id'], row['file_path']):
                                st.success(f"Deleted submission: {row['title']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete submission.")
                        
                        st.markdown("---")
                
                # View Submission Modal
                if st.session_state.view_submission is not None:
                    st.subheader("📄 Submission Details")
                    v = st.session_state.view_submission
                    st.write(f"**Student:** {v['student_name']} ({v['reg_no']})")
                    st.write(f"**Class:** {v['class']}")
                    st.write(f"**Subject:** {v['subject']}")
                    st.write(f"**Type:** {v['submission_type']}")
                    st.write(f"**Title:** {v['title']}")
                    st.write(f"**Date:** {v['date']}")
                    st.write(f"**Description:** {v['description']}")
                    st.write(f"**Points Earned:** {v['points_earned']}")
                    st.write(f"**Grade:** {v.get('grade', 'N/A')}")
                    if v.get('ai_confidence'):
                        st.write(f"**AI Confidence:** {v['ai_confidence']*100:.0f}%")
                        st.write(f"**Plagiarism Score:** {v['plagiarism_score']*100:.0f}%")
                        st.write(f"**AI Feedback:** {v['ai_feedback']}")
                    if v.get('file_path') and os.path.exists(v['file_path']):
                        st.write("**File:**")
                        dl = get_file_download_link(v['file_path'], v['file_name'])
                        if dl:
                            st.markdown(dl, unsafe_allow_html=True)
                    if st.button("Close View"):
                        st.session_state.view_submission = None
                        st.rerun()
                
                # Edit Submission Modal
                if st.session_state.edit_submission_id is not None:
                    st.subheader("✏️ Edit Submission")
                    ed = st.session_state.edit_submission_data
                    with st.form(key=f"edit_sub_form_{st.session_state.edit_submission_id}"):
                        new_title = st.text_input("Title", value=ed['title'])
                        new_description = st.text_area("Description", value=ed['description'], height=150)
                        new_points = st.number_input("Points Earned", value=float(ed['points_earned']), step=1.0)
                        new_grade = st.text_input("Grade", value=ed.get('grade', 'A'))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("💾 Save Changes"):
                                if update_submission(st.session_state.edit_submission_id, new_title, new_description, new_points, new_grade):
                                    st.success("Submission updated successfully!")
                                    st.session_state.edit_submission_id = None
                                    st.rerun()
                                else:
                                    st.error("Failed to update submission.")
                        with col2:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state.edit_submission_id = None
                                st.rerun()
                
                st.markdown("---")
                st.caption("💡 Click 👁️ to view details, ✏️ to edit, 🗑️ to delete")
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
                        
                        # View button
                        if cols[7].button("👁️", key=f"view_act_{row['activity_id']}", help="View Details"):
                            st.session_state.view_activity = row.to_dict()
                            st.rerun()
                        
                        # Edit button
                        if cols[8].button("✏️", key=f"edit_act_{row['activity_id']}", help="Edit"):
                            st.session_state.edit_activity_id = row['activity_id']
                            st.session_state.edit_activity_data = row.to_dict()
                            st.rerun()
                        
                        # Delete button
                        if cols[9].button("🗑️", key=f"del_act_{row['activity_id']}", help="Delete Activity"):
                            if delete_activity(row['activity_id'], row['file_path']):
                                st.success(f"Deleted activity: {row['topic']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete activity.")
                        
                        st.markdown("---")
                
                # View Activity Modal
                if st.session_state.view_activity is not None:
                    st.subheader("📄 Activity Details")
                    v = st.session_state.view_activity
                    st.write(f"**Student:** {v['student_name']} ({v['reg_no']})")
                    st.write(f"**Class:** {v['class']}")
                    st.write(f"**Activity Type:** {v['activity_type']}")
                    st.write(f"**Topic:** {v['topic']}")
                    st.write(f"**Date:** {v['date']}")
                    st.write(f"**Duration:** {v['duration_minutes']} minutes")
                    st.write(f"**Points Earned:** {v['points_earned']}")
                    st.write(f"**Remarks:** {v.get('remarks', 'N/A')}")
                    if v.get('file_path') and os.path.exists(v['file_path']):
                        st.write("**File:**")
                        dl = get_file_download_link(v['file_path'], v['file_name'])
                        if dl:
                            st.markdown(dl, unsafe_allow_html=True)
                    if st.button("Close View"):
                        st.session_state.view_activity = None
                        st.rerun()
                
                # Edit Activity Modal
                if st.session_state.edit_activity_id is not None:
                    st.subheader("✏️ Edit Activity")
                    ed = st.session_state.edit_activity_data
                    with st.form(key=f"edit_act_form_{st.session_state.edit_activity_id}"):
                        new_topic = st.text_input("Topic", value=ed['topic'])
                        new_remarks = st.text_area("Remarks", value=ed.get('remarks', ''), height=100)
                        new_points = st.number_input("Points Earned", value=float(ed['points_earned']), step=1.0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("💾 Save Changes"):
                                if update_activity(st.session_state.edit_activity_id, new_topic, new_remarks, new_points):
                                    st.success("Activity updated successfully!")
                                    st.session_state.edit_activity_id = None
                                    st.rerun()
                                else:
                                    st.error("Failed to update activity.")
                        with col2:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state.edit_activity_id = None
                                st.rerun()
                
                st.markdown("---")
                st.caption("💡 Click 👁️ to view details, ✏️ to edit, 🗑️ to delete")
            else:
                st.info("No extra activities submitted yet.")

    elif st.session_state.page == "🤖 AI Reference Answers":
        # ... (unchanged, same as previous)
        st.header("🤖 AI Reference Answers Management")
        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ scikit-learn not installed.")
        st.info("Add reference answers to improve AI validation.")
        tab1, tab2 = st.tabs(["➕ Add Reference Answer", "📋 View Reference Answers"])
        with tab1:
            with st.form("add_reference_form"):
                col1, col2 = st.columns(2)
                with col1:
                    subject = st.text_input("Subject*", placeholder="e.g., Database Management")
                    topic = st.text_input("Topic*", placeholder="e.g., SQL Basics")
                with col2:
                    st.info("This answer will be used for similarity checks.")
                answer_text = st.text_area("Reference Answer*", height=200)
                if st.form_submit_button("Add Reference Answer"):
                    if subject and topic and answer_text:
                        if add_reference_answer(subject, topic, answer_text, teacher_id):
                            st.success("Reference answer added!")
                            st.rerun()
                    else:
                        st.error("Please fill all fields.")
        with tab2:
            try:
                result = supabase.table('reference_answers').select('*').order('subject').order('topic').execute()
                if result.data:
                    for row in result.data:
                        with st.expander(f"📚 {row['subject']} - {row['topic']}"):
                            st.write(f"**Answer:** {row['answer_text']}")
                            st.write(f"*Added: {row['created_at']}*")
                else:
                    st.info("No reference answers yet.")
            except Exception as e:
                st.error(f"Error loading: {e}")

    elif st.session_state.page == "📊 Class Analytics":
        # ... (unchanged)
        st.header("Class Analytics")
        class_perf = supabase.table('students').select('class, total_points').execute()
        if class_perf.data:
            df = pd.DataFrame(class_perf.data)
            perf = df.groupby('class').agg(
                student_count=('class','count'),
                avg_points=('total_points','mean'),
                max_points=('total_points','max'),
                min_points=('total_points','min'),
                total_points=('total_points','sum')
            ).reset_index().sort_values('avg_points', ascending=False)
            st.subheader("📈 Class Performance")
            st.dataframe(perf, use_container_width=True)
        ai_metrics = supabase.table('submissions').select('ai_confidence, plagiarism_score').gt('ai_confidence', 0).execute()
        if ai_metrics.data:
            df_ai = pd.DataFrame(ai_metrics.data)
            st.subheader("🤖 AI Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg AI Confidence", f"{df_ai['ai_confidence'].mean()*100:.0f}%")
            col2.metric("Avg Originality", f"{(1-df_ai['plagiarism_score'].mean())*100:.0f}%")
            col3.metric("AI-Graded Submissions", len(df_ai))
        subj_dist = supabase.table('student_subjects').select('subject_id').execute()
        if subj_dist.data:
            subj_ids = [s['subject_id'] for s in subj_dist.data]
            if subj_ids:
                subjects = supabase.table('subjects').select('subject_id, subject_code, subject_name, class, teachers(name)').in_('subject_id', subj_ids).execute()
                if subjects.data:
                    df_subj = pd.DataFrame(subjects.data)
                    df_subj['student_count'] = df_subj.apply(lambda x: subj_ids.count(x['subject_id']), axis=1)
                    df_subj['teacher_name'] = df_subj['teachers'].apply(lambda x: x['name'] if x else None)
                    st.subheader("📚 Subject-wise Student Distribution")
                    st.dataframe(df_subj[['subject_code','subject_name','class','teacher_name','student_count']], use_container_width=True)

    elif st.session_state.page == "🏆 Leaderboard":
        st.header("Teacher View: Student Leaderboard")
        leaderboard = get_leaderboard(50)
        if not leaderboard.empty:
            st.dataframe(leaderboard, use_container_width=True)
        else:
            st.info("No students in leaderboard.")

    elif st.session_state.page == "👤 Edit Profile":
        # ... (unchanged)
        st.header("Edit Profile")
        with st.form("edit_teacher_profile_form"):
            name = st.text_input("Full Name", value=teacher_name)
            email = st.text_input("Email", value=teacher_email)
            department = st.text_input("Department", value=teacher_dept)
            new_pass = st.text_input("New Password (optional)", type="password")
            confirm = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Update Profile")
            if submitted:
                if new_pass:
                    if new_pass != confirm:
                        st.error("Passwords do not match!")
                    else:
                        updates = {'name': name, 'email': email, 'department': department, 'password': hash_password(new_pass)}
                        supabase.table('teachers').update(updates).eq('teacher_id', teacher_id).execute()
                        st.success("Profile updated! Please login again.")
                        st.session_state.current_teacher = None
                        st.session_state.user_role = None
                        st.session_state.logged_in = False
                        st.session_state.page = "Welcome"
                        st.rerun()
                else:
                    updates = {'name': name, 'email': email, 'department': department}
                    supabase.table('teachers').update(updates).eq('teacher_id', teacher_id).execute()
                    st.success("Profile updated successfully!")
                    st.session_state.current_teacher = supabase.table('teachers').select('*').eq('teacher_id', teacher_id).execute().data[0]
                    st.rerun()

    elif st.session_state.page == "⚙️ Manage System":
        # ... (unchanged)
        st.header("System Management")
        tab1, tab2 = st.tabs(["📊 System Stats", "⚙️ Settings"])
        with tab1:
            st.subheader("System Statistics")
            stats = {}
            stats["Total Students"] = supabase.table('students').select('student_id').execute().data.__len__()
            stats["Total Teachers"] = supabase.table('teachers').select('teacher_id').execute().data.__len__()
            stats["Total Subjects"] = supabase.table('subjects').select('subject_id').execute().data.__len__()
            stats["Total Submissions"] = supabase.table('submissions').select('submission_id').execute().data.__len__()
            stats["Total Activities"] = supabase.table('activities').select('activity_id').execute().data.__len__()
            total_points = supabase.table('students').select('total_points').execute().data
            stats["Total Points Awarded"] = sum(p['total_points'] for p in total_points) if total_points else 0
            stats["AI-Graded Submissions"] = supabase.table('submissions').select('submission_id').gt('ai_confidence', 0).execute().data.__len__()
            stats["Pending Deletion Requests"] = supabase.table('deletion_requests').select('request_id').eq('status', 'Pending').execute().data.__len__()
            df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
            st.dataframe(df_stats, use_container_width=True, hide_index=True)
            st.info("📌 Data is automatically cleaned – only last 6 months of submissions and activities are kept (via scheduled job).")
        with tab2:
            st.subheader("System Settings")
            st.success("✅ Auto-grading with AI is enabled")
            st.success("✅ Duplicate submission prevention is enabled")
            st.write("**Current Points System:**")
            st.write("- Daily Homework: 5 points (AI-adjusted)")
            st.write("- Seminar: 10 points (AI-adjusted)")
            st.write("- Project: 15 points (AI-adjusted)")
            st.write("- Extra Activity: 25 points (AI-adjusted)")
            st.write("- Weekly Assignment: 15 points (AI-adjusted)")
            st.write("- Monthly Assignment: 30 points (AI-adjusted)")
            st.write("- Research Paper: 25 points (AI-adjusted)")
            st.write("- Lab Report: 8 points (AI-adjusted)")

# ========== FOOTER TABS ==========
st.markdown("---")
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
with footer_col1:
    if st.button("🔒 Privacy Policy", use_container_width=True):
        st.session_state.show_privacy = not st.session_state.show_privacy
        for k in ['show_terms','show_contact','show_deletion']:
            st.session_state[k] = False
with footer_col2:
    if st.button("📜 Terms & Disclaimer", use_container_width=True):
        st.session_state.show_terms = not st.session_state.show_terms
        for k in ['show_privacy','show_contact','show_deletion']:
            st.session_state[k] = False
with footer_col3:
    if st.button("📩 Contact / About", use_container_width=True):
        st.session_state.show_contact = not st.session_state.show_contact
        for k in ['show_privacy','show_terms','show_deletion']:
            st.session_state[k] = False
with footer_col4:
    if st.button("🗑️ Data Deletion", use_container_width=True):
        st.session_state.show_deletion = not st.session_state.show_deletion
        for k in ['show_privacy','show_terms','show_contact']:
            st.session_state[k] = False

if st.session_state.show_privacy:
    with st.container():
        st.markdown("### 🔒 Privacy Policy")
        st.info("Privacy policy will be available soon.")
if st.session_state.show_terms:
    with st.container():
        st.markdown("### 📜 Terms & Disclaimer")
        st.info("Terms will be available soon.")
if st.session_state.show_contact:
    with st.container():
        st.markdown("### 📩 Contact & About Developer")
        st.info("Contact information will be available soon.")
if st.session_state.show_deletion:
    with st.container():
        st.markdown("### 🗑️ Data Deletion Request")
        st.warning("⚠️ This action is irreversible.")
        if st.session_state.user_role:
            user_email = st.session_state.current_student['email'] if st.session_state.user_role == "student" else st.session_state.current_teacher['email']
            st.info(f"Your email: {user_email}")
            with st.form("deletion_request_form"):
                reason = st.text_area("Reason (optional)")
                confirm = st.text_input("Type 'DELETE' to confirm", type="password")
                if st.form_submit_button("✅ Request Deletion"):
                    if confirm == "DELETE":
                        if request_data_deletion(user_email, st.session_state.user_role, reason):
                            st.success("Request submitted. Admin will process within 7 days.")
                    else:
                        st.error("Please type 'DELETE' to confirm.")
        else:
            with st.form("deletion_request_public"):
                email = st.text_input("Email*")
                user_type = st.selectbox("I am a*", ["Student","Teacher"])
                reason = st.text_area("Reason (optional)")
                confirm = st.text_input("Type 'DELETE' to confirm*", type="password")
                if st.form_submit_button("Submit Request"):
                    if email and confirm == "DELETE":
                        if request_data_deletion(email, user_type.lower(), reason):
                            st.success("Request submitted.")
                    else:
                        st.error("Please provide email and type 'DELETE'.")

# Main footer
st.markdown("""
<div style='text-align: center; padding: 15px 0; margin-top: 20px; border-top: 1px solid #ddd;'>
    <p style='margin: 5px 0; font-weight: bold;'>Continuous Student Evaluation & Monitoring System</p>
    <p style='margin: 3px 0;'>Design and Maintained by: S P Sajjan, Assistant Professor, GFGCW, Jamkhandi</p>
    <p style='margin: 3px 0;'>📧 Contact: sajjanvsl@gmail.com | 📞 Help Desk: 9008802403</p>
    <p style='margin: 5px 0;'>✅ AI-Powered Validation | 📚 Faculty Edit | 🔐 Forgot Password | 📂 File Upload/Download/View | 🚫 Duplicate Prevention | 👁️ View | ✏️ Edit | 🗑️ Delete</p>
    <p style='margin: 3px 0; color: #666; font-size: 0.9em;'>📅 Data retention: 6 months (automatic cleanup)</p>
</div>
""", unsafe_allow_html=True)
