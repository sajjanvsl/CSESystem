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

# ---------- Supabase CRUD functions (unchanged from previous working version) ----------
# ... (keep all existing CRUD functions exactly as in your last working version) ...
# For brevity, I'm not repeating them here, but you must keep them.
# They include: get_student_by_email_or_regno, add_student, edit_student_registration,
# faculty_edit_student, delete_student, get_all_students, get_student_subjects,
# register_student_subjects, remove_student_subject, add_teacher, authenticate_teacher,
# get_all_teachers, add_subject, delete_subject, get_all_subjects, assign_subject_to_teacher,
# add_submission, get_student_submissions, get_all_submissions_for_teacher, add_activity,
# get_student_activities, add_reward_claim, get_reward_history, get_daily_activity,
# update_daily_activity, get_leaderboard, get_student_progress, update_student_streak,
# validate_submission_with_ai, add_reference_answer, check_duplicate_submission,
# get_auto_grade_points, get_auto_grade_letter, add_submission_with_ai, add_extra_activity,
# forgot_password, reset_password, request_data_deletion, get_file_download_link, get_file_view_link

# ---------- FIXED check_duplicate_submission (with AI similarity) ----------
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

# ---------- FIXED add_submission_with_ai (round points to integer) ----------
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

# ---------- Teacher function to get duplicate submissions ----------
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
# (Keep all existing student pages exactly as they were from your last working version.
# I'm omitting them here for brevity, but you must include them.)
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
        # ... (keep existing dashboard code) ...
        pass

    elif st.session_state.page == "📚 Subject Management":
        # ... (keep existing subject management) ...
        pass

    elif st.session_state.page == "👨‍🎓 Manage Students":
        # ... (keep existing manage students) ...
        pass

    elif st.session_state.page == "📂 View Submissions":
        # ... (keep existing view submissions) ...
        pass

    # ---------- NEW: Duplicate Submissions Page ----------
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
        # Subject distribution (FIXED: use subject_id directly)
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
        # ... (keep existing leaderboard) ...
        pass

    elif st.session_state.page == "👤 Edit Profile":
        # ... (keep existing edit profile) ...
        pass

    elif st.session_state.page == "⚙️ Manage System":
        # ... (keep existing manage system) ...
        pass

    elif st.session_state.page == "🤖 AI Reference Answers":
        # ... (keep existing AI reference answers) ...
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
