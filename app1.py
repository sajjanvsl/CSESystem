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

# ---------- FIX: Use persistent storage location ----------
# For Streamlit Cloud, we need to store data in a persistent directory
# The /mount/src/... path is ephemeral - we need to use the app's directory
# but ensure we don't recreate the database on each run

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

# ---------- Database Helper (prevents locks) ----------
def get_db_connection():
    """Return a database connection with busy timeout and thread safety."""
    global DB_PATH
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds
    return conn

# ---------- Database Initialisation (only if needed) ----------
def init_database():
    """Initialize database only if tables don't exist."""
    conn = get_db_connection()
    c = conn.cursor()

    # Check if tables already exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    if c.fetchone():
        # Tables already exist, no need to recreate
        print("✅ Database already exists, skipping initialization")
        conn.close()
        st.session_state.db_initialized = True
        return

    print("🆕 Creating new database tables...")

    # Submissions table with AI columns
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

    # Reference answers table for AI validation
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

# ---------- Test User Creation (Only if they don't exist) ----------
def ensure_test_users():
    """Create test users only if they don't exist - preserves existing registrations."""
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

    # Add some sample reference answers only if none exist
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
    """Delete submissions and activities older than 6 months - preserves user accounts."""
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

# Run cleanup occasionally (not on every run)
if random.randint(1, 100) == 1:  # 1% chance on each run
    cleanup_old_data()

# ---------- The rest of your existing code follows ----------
# [All your existing functions and UI code remain exactly the same]
# ... (copy all your existing functions from your current code here) ...

# ==================== STREAMLIT UI ====================
st.title("📚 Continuous Student Evaluation & Monitoring System")
st.markdown("---")

# Show database location in debug mode
with st.expander("🔧 System Info", expanded=False):
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

# [Continue with your existing UI code...]
# (copy all your existing UI code from st.sidebar onwards)
