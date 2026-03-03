import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import random
import string
import os
import base64
import time
import re
import numpy as np

# Try to import sklearn, but provide fallback if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define dummy classes/functions if needed
    class TfidfVectorizer:
        def fit_transform(self, texts):
            return np.zeros((len(texts), 1))
    def cosine_similarity(a, b):
        return np.zeros((a.shape[0], b.shape[0]))

st.set_page_config(page_title="Student Evaluation System", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# ---------- FIX: Use persistent storage location ----------
import os
import tempfile

# Determine the database path based on environment
def get_db_path():
    """Get a persistent database path that works on Streamlit Cloud."""
    # For local development, use current directory
    if os.path.exists('/mount/src'):
        # We're on Streamlit Cloud - use a persistent location
        # The home directory is persistent
        home_dir = os.path.expanduser('~')
        db_dir = os.path.join(home_dir, '.student_eval_db')
    else:
        # Local development - use current directory
        db_dir = '.'
    
    # Create directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    
    return os.path.join(db_dir, 'student_evaluation.db')

# Set the database path globally
DB_PATH = get_db_path()
print(f"📁 Database location: {DB_PATH}")

# ---------- Session State for Persistence ----------
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
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# ---------- Helper Functions (MUST be defined first) ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def generate_temp_password(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

# ---------- Registration Validation ----------
def validate_class_name(class_name):
    """Validate and normalize class name to prevent inconsistent entries."""
    if not class_name or len(class_name.strip()) < 2:
        return False, "❌ Class name is too short. Please use format like 'BCA VI', 'BA II', etc."
    
    # Remove extra spaces and normalize
    class_name = ' '.join(class_name.strip().split())
    original_input = class_name
    
    # Convert to uppercase for pattern matching
    class_upper = class_name.upper()
    
    # Common patterns for class names
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
                roman_map = {'I': 'I', 'II': 'II', 'III': 'III', 'IV': 'IV', 'V': 'V', 'VI': 'VI', 
                            'VII': 'VII', 'VIII': 'VIII', 'IX': 'IX', 'X': 'X'}
                
                if suffix in roman_map:
                    normalized = f"{prefix} {roman_map[suffix]}"
                elif suffix.isdigit():
                    num = int(suffix)
                    if 1 <= num <= 10:
                        roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
                        normalized = f"{prefix} {roman_numerals[num-1]}"
                    else:
                        normalized = f"{prefix} {suffix}"
                else:
                    normalized = f"{prefix} {suffix}"
                
                normalized = ' '.join(normalized.split())
                return True, normalized
    
    if len(class_name) <= 30 and re.match(r'^[A-Za-z0-9\s\-]+$', class_name):
        return True, class_name.upper()
    
    return False, "❌ Invalid class name format. Please use standard format like 'BCA VI', 'BA II', 'BCom I', 'Semester 1', etc."

# ---------- Database Helper ----------
def get_db_connection():
    """Return a database connection with busy timeout and thread safety."""
    global DB_PATH
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn

# ---------- Database Initialisation ----------
def init_database():
    """Initialize database only if tables don't exist."""
    conn = get_db_connection()
    c = conn.cursor()

    # Check if tables already exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    if c.fetchone():
        print("✅ Database already exists, skipping initialization")
        conn.close()
        st.session_state.db_initialized = True
        return

    print("🆕 Creating new database tables...")

    # Submissions table
    c.execute('''
        CREATE TABLE submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            submission_type TEXT NOT NULL,
            subject TEXT,
            title TEXT,
            description TEXT,
            date DATE NOT NULL,
            status TEXT DEFAULT 'Submitted',
            teacher_feedback TEXT,
            grade TEXT,
            points_earned INTEGER DEFAULT 0,
            max_points INTEGER DEFAULT 50,
            file_path TEXT,
            file_name TEXT,
            file_type TEXT,
            file_size INTEGER,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            graded_at TIMESTAMP,
            graded_by INTEGER,
            auto_graded INTEGER DEFAULT 0,
            ai_confidence DECIMAL(3,2) DEFAULT 0,
            ai_feedback TEXT,
            plagiarism_score DECIMAL(3,2) DEFAULT 0
        )
    ''')

    # Students table
    c.execute('''
        CREATE TABLE students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT,
            password TEXT,
            total_points INTEGER DEFAULT 0,
            current_streak INTEGER DEFAULT 0,
            best_streak INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active DATE
        )
    ''')

    # Teachers table
    c.execute('''
        CREATE TABLE teachers (
            teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Subjects table
    c.execute('''
        CREATE TABLE subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_code TEXT UNIQUE NOT NULL,
            subject_name TEXT NOT NULL,
            class TEXT NOT NULL,
            teacher_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Student subjects junction
    c.execute('''
        CREATE TABLE student_subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            subject_id INTEGER,
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Active',
            UNIQUE(student_id, subject_id)
        )
    ''')

    # Activities table
    c.execute('''
        CREATE TABLE activities (
            activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            activity_type TEXT NOT NULL,
            topic TEXT,
            date DATE NOT NULL,
            duration_minutes INTEGER,
            points_earned INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Completed',
            remarks TEXT,
            file_path TEXT,
            file_name TEXT
        )
    ''')

    # Daily activity table
    c.execute('''
        CREATE TABLE daily_activity (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            activity_date DATE NOT NULL,
            submission_count INTEGER DEFAULT 0,
            activity_count INTEGER DEFAULT 0,
            total_points_earned INTEGER DEFAULT 0,
            study_hours DECIMAL(3,1) DEFAULT 0,
            attendance_status TEXT DEFAULT 'Present',
            remarks TEXT,
            UNIQUE(student_id, activity_date)
        )
    ''')

    # Rewards table
    c.execute('''
        CREATE TABLE rewards (
            reward_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            reward_type TEXT NOT NULL,
            points_cost INTEGER,
            reward_date DATE NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'Available',
            claimed_at TIMESTAMP
        )
    ''')

    # Point transactions table
    c.execute('''
        CREATE TABLE point_transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            transaction_type TEXT NOT NULL,
            points INTEGER NOT NULL,
            description TEXT,
            reference_id INTEGER,
            transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Password reset table
    c.execute('''
        CREATE TABLE password_reset (
            reset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            reset_code TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            used BOOLEAN DEFAULT 0
        )
    ''')

    # Reference answers table
    c.execute('''
        CREATE TABLE reference_answers (
            answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER
        )
    ''')

    # Deletion requests table
    c.execute('''
        CREATE TABLE deletion_requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            user_type TEXT NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'Pending',
            requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        )
    ''')

    # Indexes
    c.execute('CREATE INDEX idx_submissions_date ON submissions(date)')
    c.execute('CREATE INDEX idx_activities_date ON activities(date)')
    c.execute('CREATE INDEX idx_submissions_student ON submissions(student_id)')
    c.execute('CREATE INDEX idx_activities_student ON activities(student_id)')
    c.execute('CREATE INDEX idx_student_subjects ON student_subjects(student_id, subject_id)')
    c.execute('CREATE INDEX idx_reference_answers ON reference_answers(subject, topic)')

    conn.commit()
    conn.close()
    st.session_state.db_initialized = True
    print("✅ Database initialization complete")

# ---------- Test User Creation ----------
def ensure_test_users():
    """Create test users only if they don't exist."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if test student exists
    c.execute("SELECT student_id FROM students WHERE email = ?", ("test@student.com",))
    if not c.fetchone():
        password_hash = hash_password("test123")
        c.execute('''
            INSERT INTO students (reg_no, name, class, email, phone, password, last_active)
            VALUES (?, ?, ?, ?, ?, ?, DATE("now"))
        ''', ("TEST001", "Test Student", "BCA VI", "test@student.com", "1234567890", password_hash))
        print("✅ Test student created.")
    else:
        print("✅ Test student already exists")
    
    # Check if test teacher exists
    c.execute("SELECT teacher_id FROM teachers WHERE email = ?", ("test@teacher.com",))
    if not c.fetchone():
        teacher_hash = hash_password("test123")
        c.execute('''
            INSERT INTO teachers (teacher_code, name, email, password, department)
            VALUES (?, ?, ?, ?, ?)
        ''', ("T001", "Test Teacher", "test@teacher.com", teacher_hash, "Computer Science"))
        print("✅ Test teacher created.")
    else:
        print("✅ Test teacher already exists")

    # Add sample reference answers if none exist
    c.execute("SELECT COUNT(*) FROM reference_answers")
    if c.fetchone()[0] == 0:
        sample_answers = [
            ("Database Management", "SQL Basics", "SQL is a standard language for storing, manipulating and retrieving data in databases. Key commands include SELECT, INSERT, UPDATE, DELETE."),
            ("Database Management", "Normalization", "Normalization is the process of organizing data to reduce redundancy. Normal forms include 1NF, 2NF, 3NF, and BCNF."),
            ("Web Technologies", "HTML", "HTML (HyperText Markup Language) is the standard markup language for creating web pages and web applications."),
            ("Python Programming", "Variables", "Variables are containers for storing data values. Python has no command for declaring a variable."),
        ]
        for subject, topic, answer in sample_answers:
            c.execute('''
                INSERT INTO reference_answers (subject, topic, answer_text)
                VALUES (?, ?, ?)
            ''', (subject, topic, answer))
        print("✅ Sample reference answers added.")
    else:
        print("✅ Reference answers already exist")

    conn.commit()
    conn.close()

# ---------- Initialize database (only once) ----------
if not st.session_state.db_initialized:
    init_database()
    ensure_test_users()
    st.session_state.db_initialized = True

# ---------- Cleanup old data (optional, runs occasionally) ----------
def cleanup_old_data():
    """Delete submissions and activities older than 6 months."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        c.execute('DELETE FROM submissions WHERE date < ?', (six_months_ago,))
        c.execute('DELETE FROM activities WHERE date < ?', (six_months_ago,))
        c.execute('DELETE FROM daily_activity WHERE activity_date < ?', (six_months_ago,))
        c.execute('DELETE FROM password_reset WHERE created_at < ?', (six_months_ago,))
        conn.commit()
        conn.close()
        print(f"✅ Cleaned up data older than 6 months.")
    except Exception as e:
        print(f"Cleanup skipped: {e}")

# Run cleanup occasionally
if random.randint(1, 100) == 1:
    cleanup_old_data()

# ---------- Data Deletion Functions ----------
def request_data_deletion(email, user_type, reason):
    """Submit a data deletion request."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO deletion_requests (email, user_type, reason, status)
        VALUES (?, ?, ?, 'Pending')
    ''', (email, user_type, reason))
    conn.commit()
    conn.close()
    return True

def process_data_deletion(email, user_type):
    """Permanently delete user data (admin function)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    if user_type == "student":
        c.execute("SELECT student_id FROM students WHERE email = ?", (email,))
        student = c.fetchone()
        if student:
            student_id = student[0]
            c.execute('DELETE FROM submissions WHERE student_id = ?', (student_id,))
            c.execute('DELETE FROM activities WHERE student_id = ?', (student_id,))
            c.execute('DELETE FROM daily_activity WHERE student_id = ?', (student_id,))
            c.execute('DELETE FROM rewards WHERE student_id = ?', (student_id,))
            c.execute('DELETE FROM point_transactions WHERE student_id = ?', (student_id,))
            c.execute('DELETE FROM student_subjects WHERE student_id = ?', (student_id,))
            c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
    else:
        c.execute('DELETE FROM teachers WHERE email = ?', (email,))
    
    c.execute('''
        UPDATE deletion_requests 
        SET status = 'Processed', processed_at = CURRENT_TIMESTAMP
        WHERE email = ? AND status = 'Pending'
    ''', (email,))
    
    conn.commit()
    conn.close()
    return True

# ---------- AI Validation Functions ----------
def validate_submission_with_ai(submission_text, subject, topic=None):
    """Validate student submission using AI techniques."""
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
    
    conn = get_db_connection()
    c = conn.cursor()
    if topic:
        c.execute('''
            SELECT answer_text FROM reference_answers 
            WHERE subject = ? AND topic = ?
        ''', (subject, topic))
    else:
        c.execute('''
            SELECT answer_text FROM reference_answers 
            WHERE subject = ?
        ''', (subject,))
    references = [row[0] for row in c.fetchall()]
    conn.close()
    
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
        'Database Management': ['sql', 'query', 'table', 'database', 'normalization', 'index', 'data', 'server'],
        'Web Technologies': ['html', 'css', 'javascript', 'web', 'browser', 'server', 'client', 'http'],
        'Python Programming': ['python', 'variable', 'function', 'class', 'loop', 'list', 'dict', 'import'],
        'General': ['example', 'explain', 'define', 'describe', 'compare', 'analyze', 'discuss']
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
    """Add a reference answer for AI validation."""
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO reference_answers (subject, topic, answer_text, created_by)
            VALUES (?, ?, ?, ?)
        ''', (subject, topic, answer_text, teacher_id))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding reference answer: {str(e)}")
        return False
    finally:
        conn.close()

# ---------- Duplicate Submission Check ----------
def check_duplicate_submission(student_id, subject, title, description, submission_type):
    """Check if a student has already submitted a similar assignment."""
    conn = get_db_connection()
    c = conn.cursor()
    
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    c.execute('''
        SELECT submission_id, date, title FROM submissions 
        WHERE student_id = ? AND subject = ? AND title = ? AND date >= ?
        ORDER BY date DESC
    ''', (student_id, subject, title, thirty_days_ago))
    
    exact_match = c.fetchone()
    if exact_match:
        conn.close()
        return True, f"You have already submitted an assignment with the same title on {exact_match[1]}. Please use a different title."
    
    if len(description) > 50:
        c.execute('''
            SELECT submission_id, description, date FROM submissions 
            WHERE student_id = ? AND subject = ? AND date >= ?
            ORDER BY date DESC LIMIT 5
        ''', (student_id, subject, thirty_days_ago))
        
        recent_subs = c.fetchall()
        conn.close()
        
        for sub in recent_subs:
            if sub[1] and len(sub[1]) > 50:
                if SKLEARN_AVAILABLE:
                    try:
                        vectorizer = TfidfVectorizer(stop_words='english')
                        texts = [description, sub[1]]
                        tfidf = vectorizer.fit_transform(texts)
                        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                        
                        if similarity > 0.85:
                            return True, f"⚠️ This submission is very similar ({similarity:.1%}) to your submission from {sub[2]}. Please ensure you're submitting new work."
                    except:
                        pass
                else:
                    words1 = set(description.lower().split())
                    words2 = set(sub[1].lower().split())
                    if len(words1) > 0 and len(words2) > 0:
                        overlap = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = overlap / union if union > 0 else 0
                        if similarity > 0.8:
                            return True, f"⚠️ This submission has high word overlap ({similarity:.1%}) with your submission from {sub[2]}. Please ensure you're submitting new work."
    else:
        conn.close()
    
    return False, ""

# ---------- Student Functions ----------
def add_student_with_password(reg_no, name, class_name, email, password, phone=None):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        reg_no = reg_no.strip()
        email = email.strip().lower()
        
        is_valid, normalized_class = validate_class_name(class_name)
        if not is_valid:
            st.error(normalized_class)
            return False
        
        password_hash = hash_password(password)
        
        c.execute("SELECT reg_no, email FROM students WHERE reg_no = ? COLLATE NOCASE OR email = ?", (reg_no, email))
        existing = c.fetchone()
        if existing:
            st.error("Registration number or email already exists!")
            return False

        c.execute('''
            INSERT INTO students (reg_no, name, class, email, phone, password, last_active)
            VALUES (?, ?, ?, ?, ?, ?, DATE("now"))
        ''', (reg_no, name, normalized_class, email, phone, password_hash))
        conn.commit()
        st.success(f"Registration successful! Your class has been set to: {normalized_class}")
        return True
    except sqlite3.IntegrityError as e:
        st.error(f"Registration failed: {str(e)}")
        return False
    finally:
        conn.close()

def authenticate_student(login_id, password, use_regno=False):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("PRAGMA table_info(students)")
    columns = [col[1] for col in c.fetchall()]
    
    if use_regno:
        c.execute("SELECT * FROM students WHERE reg_no = ?", (login_id.strip(),))
    else:
        c.execute("SELECT * FROM students WHERE email = ?", (login_id.strip().lower(),))
    student = c.fetchone()
    conn.close()
    
    if student:
        student_dict = dict(zip(columns, student))
        stored_hash = student_dict['password']
        if stored_hash == hash_password(password):
            st.session_state.logged_in = True
            return student
    return None

def get_student(reg_no):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE reg_no = ?", (reg_no.strip(),))
    student = c.fetchone()
    conn.close()
    return student

# ---------- Teacher Functions ----------
def register_teacher_with_password(teacher_code, name, email, password, department):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        pwd_hash = hash_password(password)
        email = email.strip().lower()
        teacher_code = teacher_code.strip()
        c.execute('''
            INSERT INTO teachers (teacher_code, name, email, password, department)
            VALUES (?, ?, ?, ?, ?)
        ''', (teacher_code, name, email, pwd_hash, department))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_teacher(email, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("PRAGMA table_info(teachers)")
    columns = [col[1] for col in c.fetchall()]
    c.execute("SELECT * FROM teachers WHERE email = ?", (email.strip().lower(),))
    teacher = c.fetchone()
    conn.close()
    if teacher:
        teacher_dict = dict(zip(columns, teacher))
        stored_hash = teacher_dict['password']
        if stored_hash == hash_password(password):
            st.session_state.logged_in = True
            return teacher
    return None

def get_all_teachers():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query('SELECT teacher_id, teacher_code, name, email, department FROM teachers ORDER BY name', conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

# ---------- Subject Functions ----------
def add_subject(subject_code, subject_name, class_name, teacher_id=None):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        subject_code = subject_code.strip().upper()
        
        c.execute("SELECT subject_id FROM subjects WHERE subject_code = ? COLLATE NOCASE", (subject_code,))
        if c.fetchone():
            st.error(f"❌ Subject code '{subject_code}' already exists! Please use a different code.")
            return False
        
        is_valid, normalized_class = validate_class_name(class_name)
        if not is_valid:
            st.error(normalized_class)
            return False
            
        c.execute('''
            INSERT INTO subjects (subject_code, subject_name, class, teacher_id)
            VALUES (?, ?, ?, ?)
        ''', (subject_code, subject_name, normalized_class, teacher_id))
        conn.commit()
        st.success(f"✅ Subject '{subject_code}' created successfully for class: {normalized_class}")
        return True
    except sqlite3.IntegrityError:
        st.error(f"❌ Subject code already exists!")
        return False
    finally:
        conn.close()

def delete_subject(subject_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM student_subjects WHERE subject_id = ?", (subject_id,))
        count = c.fetchone()[0]
        
        if count > 0:
            if not st.session_state.get(f'confirm_delete_{subject_id}', False):
                st.session_state[f'confirm_delete_{subject_id}'] = True
                st.warning(f"⚠️ This subject has {count} student registrations. Delete anyway?")
                return False
        
        c.execute("DELETE FROM student_subjects WHERE subject_id = ?", (subject_id,))
        c.execute("DELETE FROM subjects WHERE subject_id = ?", (subject_id,))
        conn.commit()
        st.success("✅ Subject deleted successfully!")
        return True
    except Exception as e:
        st.error(f"Error deleting subject: {str(e)}")
        return False
    finally:
        conn.close()

def get_all_subjects(class_name=None):
    conn = get_db_connection()
    try:
        if class_name:
            query = '''
                SELECT s.*, t.name as teacher_name
                FROM subjects s
                LEFT JOIN teachers t ON s.teacher_id = t.teacher_id
                WHERE s.class = ?
                ORDER BY s.subject_name
            '''
            df = pd.read_sql_query(query, conn, params=(class_name,))
            
            if df.empty:
                is_valid, normalized_class = validate_class_name(class_name)
                if is_valid and normalized_class != class_name:
                    df = pd.read_sql_query(query, conn, params=(normalized_class,))
            
            if df.empty:
                query_ci = '''
                    SELECT s.*, t.name as teacher_name
                    FROM subjects s
                    LEFT JOIN teachers t ON s.teacher_id = t.teacher_id
                    WHERE LOWER(s.class) = LOWER(?)
                    ORDER BY s.subject_name
                '''
                df = pd.read_sql_query(query_ci, conn, params=(class_name,))
        else:
            query = '''
                SELECT s.*, t.name as teacher_name
                FROM subjects s
                LEFT JOIN teachers t ON s.teacher_id = t.teacher_id
                ORDER BY s.class, s.subject_name
            '''
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error in get_all_subjects: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def assign_subject_to_teacher(subject_id, teacher_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('UPDATE subjects SET teacher_id = ? WHERE subject_id = ?', (teacher_id, subject_id))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def register_student_subjects(student_id, subject_ids):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        for sid in subject_ids:
            c.execute('''
                INSERT OR IGNORE INTO student_subjects (student_id, subject_id, status)
                VALUES (?, ?, 'Active')
            ''', (student_id, sid))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error registering subjects: {e}")
        return False
    finally:
        conn.close()

def get_student_subjects(student_id):
    conn = get_db_connection()
    try:
        query = '''
            SELECT s.subject_id, s.subject_code, s.subject_name, s.class,
                   t.name as teacher_name, ss.registration_date
            FROM student_subjects ss
            JOIN subjects s ON ss.subject_id = s.subject_id
            LEFT JOIN teachers t ON s.teacher_id = t.teacher_id
            WHERE ss.student_id = ? AND ss.status = 'Active'
            ORDER BY s.subject_name
        '''
        df = pd.read_sql_query(query, conn, params=(student_id,))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def remove_student_subject(student_id, subject_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT id FROM student_subjects 
            WHERE student_id = ? AND subject_id = ?
        ''', (student_id, subject_id))
        
        if not c.fetchone():
            conn.close()
            st.error("Subject not found in your registration.")
            return False
        
        c.execute('''
            DELETE FROM student_subjects 
            WHERE student_id = ? AND subject_id = ?
        ''', (student_id, subject_id))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error removing subject: {str(e)}")
        return False
    finally:
        conn.close()

# ---------- Forgot Password ----------
def forgot_password(email, user_type):
    email = email.strip().lower()
    conn = get_db_connection()
    c = conn.cursor()

    try:
        if user_type == "student":
            c.execute("SELECT student_id, name FROM students WHERE email = ?", (email,))
        else:
            c.execute("SELECT teacher_id, name FROM teachers WHERE email = ?", (email,))

        user = c.fetchone()

        if not user:
            conn.close()
            return False, "Email not found in our records."

        temp_password = generate_temp_password(8)
        password_hash = hash_password(temp_password)

        if user_type == "student":
            c.execute("UPDATE students SET password = ? WHERE email = ?", (password_hash, email))
        else:
            c.execute("UPDATE teachers SET password = ? WHERE email = ?", (password_hash, email))

        expires_at = datetime.now() + timedelta(hours=24)
        c.execute('''
            INSERT INTO password_reset (email, reset_code, expires_at)
            VALUES (?, ?, ?)
        ''', (email, temp_password, expires_at))

        conn.commit()
        conn.close()
        return True, temp_password

    except Exception as e:
        conn.close()
        return False, f"An error occurred: {str(e)}"

def reset_password(email, new_password):
    conn = get_db_connection()
    c = conn.cursor()
    email = email.strip().lower()
    c.execute("SELECT student_id FROM students WHERE email = ?", (email,))
    student = c.fetchone()
    if student:
        pwd_hash = hash_password(new_password)
        c.execute("UPDATE students SET password = ? WHERE email = ?", (pwd_hash, email))
        conn.commit()
        conn.close()
        return True, "student"

    c.execute("SELECT teacher_id FROM teachers WHERE email = ?", (email,))
    teacher = c.fetchone()
    if teacher:
        pwd_hash = hash_password(new_password)
        c.execute("UPDATE teachers SET password = ? WHERE email = ?", (pwd_hash, email))
        conn.commit()
        conn.close()
        return True, "teacher"

    conn.close()
    return False, None

# ---------- File Handling ----------
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

# ---------- Auto‑grading Functions ----------
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
    
    is_duplicate, duplicate_msg = check_duplicate_submission(student_id, subject, title, description, submission_type)
    if is_duplicate:
        st.error(duplicate_msg)
        return None
    
    ai_result = validate_submission_with_ai(description, subject)
    
    if ai_result['plagiarism_score'] > 0.7:
        ai_result['feedback'] += "\n\n⚠️ **Warning:** High similarity with previous submissions detected."
    
    adjusted_points = points * ai_result['confidence']
    
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO submissions (
                student_id, submission_type, subject, title, description,
                date, file_path, file_name, file_type, file_size,
                max_points, points_earned, grade,
                status, auto_graded, graded_at,
                ai_confidence, ai_feedback, plagiarism_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Graded', 1, CURRENT_TIMESTAMP, ?, ?, ?)
        ''', (student_id, submission_type, subject, title, description,
              date, file_path, file_name, file_type, file_size,
              points, adjusted_points, grade,
              ai_result['confidence'], ai_result['feedback'], ai_result['plagiarism_score']))

        submission_id = c.lastrowid

        c.execute('UPDATE students SET total_points = total_points + ?, last_active = DATE("now") WHERE student_id = ?',
                 (adjusted_points, student_id))

        c.execute('SELECT last_active FROM students WHERE student_id = ?', (student_id,))
        result = c.fetchone()
        if result:
            last_active = result[0]
            if last_active == str(date):
                c.execute('UPDATE students SET current_streak = current_streak + 1 WHERE student_id = ?', (student_id,))
                c.execute('UPDATE students SET best_streak = MAX(best_streak, current_streak) WHERE student_id = ?', (student_id,))
            else:
                c.execute('UPDATE students SET current_streak = 1 WHERE student_id = ?', (student_id,))

        c.execute('''
            INSERT INTO point_transactions (student_id, transaction_type, points, description, reference_id)
            VALUES (?, 'Auto Graded', ?, ?, ?)
        ''', (student_id, adjusted_points, f"AI-graded: {submission_type} (Conf: {ai_result['confidence']})", submission_id))

        conn.commit()
        
        st.session_state.submission_review = ai_result
        
        return submission_id
    except Exception as e:
        st.error(f"Error adding submission: {str(e)}")
        return None
    finally:
        conn.close()

def add_extra_activity(student_id, activity_type, topic, date, duration, remarks,
                       file_path=None, file_name=None):
    points = 25
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO activities (student_id, activity_type, topic, date, duration_minutes,
                                     remarks, points_earned, file_path, file_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_id, 'Extra Activity', topic, date, duration, remarks, points, file_path, file_name))

        c.execute('UPDATE students SET total_points = total_points + ? WHERE student_id = ?', (points, student_id))
        c.execute('''
            INSERT INTO point_transactions (student_id, transaction_type, points, description)
            VALUES (?, 'Extra Activity', ?, ?)
        ''', (student_id, points, f"Extra Activity: {topic}"))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding activity: {str(e)}")
        return False
    finally:
        conn.close()

# ---------- Query Functions ----------
def get_all_students():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query('''
            SELECT student_id, reg_no, name, class, email, phone, total_points,
                   current_streak, best_streak, last_active
            FROM students ORDER BY total_points DESC
        ''', conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_student_submissions(student_id):
    conn = get_db_connection()
    try:
        df = pd.read_sql_query('''
            SELECT submission_id, submission_type, subject, title, description, date, status,
                   points_earned, max_points, grade, teacher_feedback, graded_at,
                   file_path, file_name, file_type, file_size,
                   ai_confidence, ai_feedback, plagiarism_score
            FROM submissions WHERE student_id = ? ORDER BY date DESC
        ''', conn, params=(student_id,))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_all_submissions_for_teacher():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query('''
            SELECT s.submission_id, s.submission_type, s.subject, s.title, s.date,
                   s.file_path, s.file_name, s.file_type, s.file_size,
                   s.ai_confidence, s.ai_feedback, s.plagiarism_score,
                   st.name as student_name, st.reg_no, st.class
            FROM submissions s
            JOIN students st ON s.student_id = st.student_id
            ORDER BY s.date DESC
        ''', conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_daily_activity(student_id, days=7):
    conn = get_db_connection()
    try:
        df = pd.read_sql_query('''
            SELECT activity_date, submission_count, activity_count, total_points_earned
            FROM daily_activity
            WHERE student_id = ? AND activity_date >= DATE("now", ?)
            ORDER BY activity_date DESC
        ''', conn, params=(student_id, f'-{days} days'))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_leaderboard(limit=20, class_filter=None):
    conn = get_db_connection()
    try:
        query = '''SELECT s.reg_no, s.name, s.class, s.total_points, s.current_streak, s.best_streak,
                  (SELECT COUNT(*) FROM submissions WHERE student_id = s.student_id) as submissions_total,
                  (SELECT COUNT(*) FROM activities WHERE student_id = s.student_id) as activities_count
                  FROM students s'''
        params = []
        if class_filter and class_filter != "All Classes":
            query += " WHERE s.class = ?"
            params.append(class_filter)
        query += ' ORDER BY s.total_points DESC, s.current_streak DESC LIMIT ?'
        params.append(limit)
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df.insert(0, 'Rank', range(1, len(df) + 1))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_student_progress(student_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute('''
            SELECT COUNT(*) as total_submissions, SUM(points_earned) as total_points_earned
            FROM submissions WHERE student_id = ?
        ''', (student_id,))
        subs = c.fetchone() or (0, 0)
        c.execute('''
            SELECT COUNT(*) as total_activities, SUM(points_earned) as activity_points
            FROM activities WHERE student_id = ?
        ''', (student_id,))
        acts = c.fetchone() or (0, 0)
        return {
            'total_submissions': subs[0] or 0,
            'submission_points': subs[1] or 0,
            'total_activities': acts[0] or 0,
            'activity_points': acts[1] or 0
        }
    except:
        return {'total_submissions': 0, 'submission_points': 0, 'total_activities': 0, 'activity_points': 0}
    finally:
        conn.close()

# ---------- Create uploads directory ----------
Path("uploads").mkdir(exist_ok=True)

# ==================== STREAMLIT UI ====================
st.title("📚 Continuous Student Evaluation & Monitoring System")
st.markdown("---")

# Show sklearn availability warning
if not SKLEARN_AVAILABLE:
    st.sidebar.warning("⚠️ Advanced AI features limited. Install scikit-learn for full functionality.")

# Show database location
with st.sidebar.expander("🔧 System Info", expanded=False):
    st.write(f"📁 Database location: `{DB_PATH}`")
    st.write(f"📊 Database exists: {os.path.exists(DB_PATH)}")
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM students")
        student_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM teachers")
        teacher_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM subjects")
        subject_count = c.fetchone()[0]
        conn.close()
        st.write(f"👥 Students: {student_count}")
        st.write(f"👨‍🏫 Teachers: {teacher_count}")
        st.write(f"📚 Subjects: {subject_count}")
    except Exception as e:
        st.write(f"Error reading database: {e}")

# Sidebar
with st.sidebar:
    if st.session_state.user_role:
        if st.session_state.user_role == "student":
            student = st.session_state.current_student
            if student:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("PRAGMA table_info(students)")
                columns = [col[1] for col in c.fetchall()]
                conn.close()
                student_dict = dict(zip(columns, student))
                st.header("🎓 Student Info")
                st.success(f"**{student_dict.get('name', '')}**")
                st.info(f"Reg No: {student_dict.get('reg_no', '')}")
                st.info(f"Class: {student_dict.get('class', '')}")
                st.info(f"Email: {student_dict.get('email', '')}")
                st.info(f"Points: {student_dict.get('total_points', 0)} 🏆")
                st.info(f"Streak: {student_dict.get('current_streak', 0)} days 🔥")
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
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("PRAGMA table_info(teachers)")
                columns = [col[1] for col in c.fetchall()]
                conn.close()
                teacher_dict = dict(zip(columns, teacher))
                st.header("👨‍🏫 Teacher Info")
                st.success(f"**Prof. {teacher_dict.get('name', '')}**")
                st.info(f"Email: {teacher_dict.get('email', '')}")
                st.info(f"Dept: {teacher_dict.get('department', '')}")
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
                        student = authenticate_student(email, password, use_regno=False)
                        if student:
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
                        student = authenticate_student(reg_no, password, use_regno=True)
                        if student:
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
                            st.info(f"🔑 **Your temporary password is:** `{result}`")
                            st.warning("Please copy this password and use it to log in.")
                        else:
                            st.error(f"❌ {result}")

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

# ========== MAIN CONTENT ==========
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
                confirm_password = st.text_input("Confirm Password*", type="password")
                
                if st.form_submit_button("Register"):
                    if reg_no and name and class_name and email and password:
                        if password == confirm_password:
                            if add_student_with_password(reg_no, name, class_name, email, password, phone):
                                st.success("✅ Registration successful! Please login.")
                                st.info("📚 After login, you can select your subjects.")
                            else:
                                st.error("Registration failed! Email or Registration number may already exist.")
                        else:
                            st.error("Passwords do not match!")
                    else:
                        st.error("Please fill all required fields (*)")
    
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
                t_dept = st.text_input("Department*", placeholder="e.g., History, Mathematics, Computer Science")
                
                if st.form_submit_button("Register as Teacher"):
                    if all([t_code, t_name, t_email, t_password, t_confirm, t_dept]):
                        if t_password == t_confirm:
                            if register_teacher_with_password(t_code, t_name, t_email, t_password, t_dept):
                                st.success("✅ Registration successful! Please login.")
                                st.info("📚 After login, you can create and manage subjects and AI reference answers.")
                            else:
                                st.error("Teacher code or email already exists!")
                        else:
                            st.error("Passwords do not match!")
                    else:
                        st.error("Please fill all fields")

# ========== FOOTER WITH FOUR TABS ==========
st.markdown("---")

footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    if st.button("🔒 Privacy Policy", use_container_width=True):
        st.session_state.show_privacy = not st.session_state.get('show_privacy', False)
        st.session_state.show_terms = False
        st.session_state.show_contact = False
        st.session_state.show_deletion = False

with footer_col2:
    if st.button("📜 Terms & Disclaimer", use_container_width=True):
        st.session_state.show_terms = not st.session_state.get('show_terms', False)
        st.session_state.show_privacy = False
        st.session_state.show_contact = False
        st.session_state.show_deletion = False

with footer_col3:
    if st.button("📩 Contact / About", use_container_width=True):
        st.session_state.show_contact = not st.session_state.get('show_contact', False)
        st.session_state.show_privacy = False
        st.session_state.show_terms = False
        st.session_state.show_deletion = False

with footer_col4:
    if st.button("🗑️ Data Deletion", use_container_width=True):
        st.session_state.show_deletion = not st.session_state.get('show_deletion', False)
        st.session_state.show_privacy = False
        st.session_state.show_terms = False
        st.session_state.show_contact = False

st.markdown("<br>", unsafe_allow_html=True)

# Privacy Policy Tab
if st.session_state.get('show_privacy', False):
    with st.container():
        st.markdown("<h3 style='color: #1f77b4;'>🔒 Privacy Policy</h3>", unsafe_allow_html=True)
        st.info("📄 Privacy policy will be available soon.")

# Terms & Disclaimer Tab
if st.session_state.get('show_terms', False):
    with st.container():
        st.markdown("<h3 style='color: #ff6b4a;'>📜 Terms & Disclaimer</h3>", unsafe_allow_html=True)
        st.info("📄 Terms and disclaimer will be available soon.")

# Contact / About Tab
if st.session_state.get('show_contact', False):
    with st.container():
        st.markdown("<h3 style='color: #2ecc71;'>📩 Contact & About Developer</h3>", unsafe_allow_html=True)
        st.info("📄 Contact information will be available soon.")

# Data Deletion Tab
if st.session_state.get('show_deletion', False):
    with st.container():
        st.markdown("<h3 style='color: #e74c3c;'>🗑️ Data Deletion Request</h3>", unsafe_allow_html=True)
        st.info("📄 Data deletion request feature will be available soon.")

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
