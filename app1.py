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
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "sb_publishable_yfvTRBgiBGYHWgrX2ZbzVw_KLTk0ddh")   # Use the publishable key here
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

def delete_student(student_id):
    try:
        supabase.table('submissions').delete().eq('student_id', student_id).execute()
        supabase.table('activities').delete().eq('student_id', student_id).execute()
        supabase.table('daily_activity').delete().eq('student_id', student_id).execute()
        supabase.table('rewards').delete().eq('student_id', student_id).execute()
        supabase.table('point_transactions').delete().eq('student_id', student_id).execute()
        supabase.table('student_subjects').delete().eq('student_id', student_id).execute()
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
                'submission_type': s['submission_type'],
                'subject': s['subject'],
                'title': s['title'],
                'date': s['date'],
                'file_path': s['file_path'],
                'file_name': s['file_name'],
                'file_type': s['file_type'],
                'file_size': s['file_size'],
                'ai_confidence': s['ai_confidence'],
                'ai_feedback': s['ai_feedback'],
                'plagiarism_score': s['plagiarism_score'],
                'student_name': s['students']['name'],
                'reg_no': s['students']['reg_no'],
                'class': s['students']['class']
            })
        return pd.DataFrame(rows)
    except Exception as e:
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
        query = supabase.table('students').select('reg_no, name, class, total_points, current_streak, best_streak')
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

# ---------- Enhanced duplicate check ----------
def check_duplicate_submission(student_id, subject, title, description, submission_type):
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    # Exact title match
    res = supabase.table('submissions').select('submission_id, date, title').eq('student_id', student_id).eq('subject', subject).eq('title', title).gte('date', thirty_days_ago).execute()
    if res.data:
        return True, f"You have already submitted an assignment with the same title on {res.data[0]['date']}. Please use a different title."
    # AI similarity check (if scikit-learn is available)
    if SKLEARN_AVAILABLE and len(description) > 100:
        recent = supabase.table('submissions').select('submission_id, description, date').eq('student_id', student_id).eq('subject', subject).gte('date', thirty_days_ago).execute()
        if recent.data:
            texts = [description] + [r['description'] for r in recent.data if r['description']]
            if len(texts) > 1:
                try:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf = vectorizer.fit_transform(texts)
                    similarities = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
                    for i, sim in enumerate(similarities):
                        if sim > 0.85:
                            return True, f"⚠️ This submission is very similar ({sim:.1%}) to your submission from {recent.data[i]['date']}. Please submit new work."
                except:
                    pass
    return False, ""

# ---------- Fixed add_submission_with_ai (round points) ----------
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
    adjusted_points = round(adjusted_points)   # <-- FIX: convert to integer
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

# ---------- Duplicate management functions ----------
def get_duplicate_submissions():
    """Return a DataFrame of duplicate submissions (same title and subject by same student)."""
    all_subs = supabase.table('submissions').select('*, students(name, reg_no, class)').execute()
    if not all_subs.data:
        return pd.DataFrame()
    df = pd.DataFrame(all_subs.data)
    # Group by student_id, subject, title
    dup_groups = df.groupby(['student_id', 'subject', 'title']).filter(lambda x: len(x) > 1)
    if dup_groups.empty:
        return pd.DataFrame()
    # Add student name and class
    dup_groups['student_name'] = dup_groups['students'].apply(lambda x: x['name'])
    dup_groups['reg_no'] = dup_groups['students'].apply(lambda x: x['reg_no'])
    dup_groups['class'] = dup_groups['students'].apply(lambda x: x['class'])
    return dup_groups[['submission_id', 'student_name', 'reg_no', 'class', 'subject', 'title', 'date', 'points_earned']]

def delete_submission(submission_id):
    try:
        # Delete associated file if exists
        sub = supabase.table('submissions').select('file_path, points_earned, student_id').eq('submission_id', submission_id).execute()
        if sub.data:
            if sub.data[0]['file_path'] and os.path.exists(sub.data[0]['file_path']):
                os.remove(sub.data[0]['file_path'])
            # Deduct points from student
            student_id = sub.data[0]['student_id']
            points = sub.data[0]['points_earned']
            student = supabase.table('students').select('total_points').eq('student_id', student_id).execute().data[0]
            supabase.table('students').update({'total_points': student['total_points'] - points}).eq('student_id', student_id).execute()
            supabase.table('point_transactions').insert({
                'student_id': student_id,
                'transaction_type': 'Duplicate Removed',
                'points': -points,
                'description': f"Duplicate submission removed"
            }).execute()
        # Delete submission
        supabase.table('submissions').delete().eq('submission_id', submission_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting submission: {e}")
        return False

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
                "📂 View Submissions", "🚫 Duplicate Submissions", "📊 Class Analytics",
                "🏆 Leaderboard", "👤 Edit Profile", "⚙️ Manage System", "🤖 AI Reference Answers"
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
# (Keep your existing student section code here – all student pages like Dashboard, My Subjects, etc.)
# I'm omitting them for brevity, but you must keep them exactly as they were.
# ...

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
        # ... (keep your existing teacher dashboard code) ...
        pass

    elif st.session_state.page == "📚 Subject Management":
        # ... (keep your existing subject management code) ...
        pass

    elif st.session_state.page == "👨‍🎓 Manage Students":
        # ... (keep your existing manage students code) ...
        pass

    elif st.session_state.page == "📂 View Submissions":
        # ... (keep your existing view submissions code) ...
        pass

    elif st.session_state.page == "🚫 Duplicate Submissions":
        st.header("Manage Duplicate Submissions")
        st.info("Here you can view and delete duplicate submissions (same title and subject by the same student).")

        dup_df = get_duplicate_submissions()
        if not dup_df.empty:
            st.warning(f"Found {len(dup_df)} duplicate submissions across {dup_df['student_name'].nunique()} students.")
            st.dataframe(dup_df, use_container_width=True)

            st.subheader("Delete Duplicate Submissions")
            submission_ids = dup_df['submission_id'].tolist()
            selected_id = st.selectbox("Select submission to delete", submission_ids, format_func=lambda x: f"ID {x} - {dup_df[dup_df['submission_id']==x]['student_name'].iloc[0]} - {dup_df[dup_df['submission_id']==x]['title'].iloc[0]}")
            if st.button("🗑️ Delete Selected Submission", type="secondary"):
                if delete_submission(selected_id):
                    st.success(f"Submission {selected_id} deleted successfully!")
                    st.rerun()
        else:
            st.success("No duplicate submissions found. Great!")

    elif st.session_state.page == "📊 Class Analytics":
        st.header("Class Analytics")
        # Class performance
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
        # AI metrics
        ai_metrics = supabase.table('submissions').select('ai_confidence, plagiarism_score').gt('ai_confidence', 0).execute()
        if ai_metrics.data:
            df_ai = pd.DataFrame(ai_metrics.data)
            st.subheader("🤖 AI Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg AI Confidence", f"{df_ai['ai_confidence'].mean()*100:.0f}%")
            col2.metric("Avg Originality", f"{(1-df_ai['plagiarism_score'].mean())*100:.0f}%")
            col3.metric("AI-Graded Submissions", len(df_ai))
        # Subject distribution (FIXED)
        subj_dist = supabase.table('student_subjects').select('subject_id').execute()
        if subj_dist.data:
            subj_ids = [s['subject_id'] for s in subj_dist.data]
            if subj_ids:
                subjects = supabase.table('subjects').select('subject_id, subject_code, subject_name, class, teachers(name)').in_('subject_id', subj_ids).execute()
                if subjects.data:
                    df_subj = pd.DataFrame(subjects.data)
                    df_subj['student_count'] = df_subj['subject_id'].apply(lambda x: subj_ids.count(x))
                    df_subj['teacher_name'] = df_subj['teachers'].apply(lambda x: x['name'] if x else None)
                    st.subheader("📚 Subject-wise Student Distribution")
                    st.dataframe(df_subj[['subject_code','subject_name','class','teacher_name','student_count']], use_container_width=True)

    elif st.session_state.page == "🏆 Leaderboard":
        # ... (keep your existing leaderboard code) ...
        pass

    elif st.session_state.page == "👤 Edit Profile":
        # ... (keep your existing edit profile code) ...
        pass

    elif st.session_state.page == "⚙️ Manage System":
        # ... (keep your existing manage system code) ...
        pass

    elif st.session_state.page == "🤖 AI Reference Answers":
        # ... (keep your existing AI reference answers code) ...
        pass

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
    <p style='margin: 5px 0;'>✅ AI-Powered Validation | 📚 Faculty Edit | 🔐 Forgot Password | 📂 File Upload/Download/View | 🚫 Duplicate Prevention</p>
    <p style='margin: 3px 0; color: #666; font-size: 0.9em;'>📅 Data retention: 6 months (automatic cleanup)</p>
</div>
""", unsafe_allow_html=True)
