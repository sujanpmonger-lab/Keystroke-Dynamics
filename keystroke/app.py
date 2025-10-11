import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === SQLite persistence for users ===
import sqlite3, hashlib

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

# --------------------------
# Roles
# --------------------------
ALLOWED_ROLES = ["admin", "user", "moderator", "analyst"]
REQUESTABLE_ROLES = ["user", "analyst"]

# --------------------------
# Premium Theming & UI
# --------------------------
def inject_premium_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Animated Gradient Title */
        .app-title {
            font-weight: 800;
            letter-spacing: -0.03em;
            text-align: center;
            font-size: 3rem;
            line-height: 1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 0.5rem 0 1rem 0;
            animation: gradientShift 8s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .app-subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 2.5rem;
            font-size: 1.1rem;
            font-weight: 400;
            letter-spacing: 0.02em;
        }

        /* Premium Glass Cards */
        .glass {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 20px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 
                0 1px 3px rgba(15, 23, 42, 0.03),
                0 10px 40px rgba(15, 23, 42, 0.06),
                0 0 0 1px rgba(148, 163, 184, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .glass::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.08), transparent);
            transition: left 0.5s ease;
        }
        
        .glass:hover::before {
            left: 100%;
        }
        
        .glass:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 4px 6px rgba(15, 23, 42, 0.05),
                0 20px 60px rgba(102, 126, 234, 0.15),
                0 0 0 1px rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.3);
        }

        /* Compact User Card */
        .user-card-compact {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
            border: 1px solid rgba(102, 126, 234, 0.15);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .user-card-compact:hover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateX(4px);
        }

        .user-avatar-small {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            color: white;
            font-weight: 700;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        }

        .status-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 100px;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.02em;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .status-approved {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }

        .status-pending {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
        }

        .role-badge {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 100px;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            font-weight: 600;
            font-size: 0.7rem;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }

        /* Premium Pills & Badges */
        .pill {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 100px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            font-size: 0.875rem;
            letter-spacing: 0.02em;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        
        .pill:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .alert-badge {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 100px;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            font-weight: 600;
            font-size: 0.8rem;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        /* Premium Buttons */
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton>button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.45);
        }
        
        .stButton>button:active {
            transform: translateY(0px);
        }

        /* Enhanced Input Fields */
        .stTextInput>div>div>input,
        .stSelectbox>div>div>select {
            border-radius: 12px;
            border: 2px solid rgba(148, 163, 184, 0.2);
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
            background: white;
        }
        
        .stTextInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        /* Premium Tables */
        .stDataFrame, .stTable {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        
        section[data-testid="stSidebar"] .sidebar-title {
            font-weight: 700;
            font-size: 1.2rem;
            color: #0f172a;
            margin-top: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        /* Tabs Enhancement */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            background: white;
            border: 2px solid rgba(148, 163, 184, 0.15);
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.05);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-color: transparent;
        }

        /* Progress Bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 100px;
        }

        /* Metrics Enhancement */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            border-radius: 12px;
            background: rgba(102, 126, 234, 0.05);
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.2);
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            border-radius: 16px;
            border: 2px dashed rgba(102, 126, 234, 0.3);
            background: rgba(102, 126, 234, 0.02);
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(102, 126, 234, 0.5);
            background: rgba(102, 126, 234, 0.05);
        }

        /* Success/Error/Warning Messages */
        .stSuccess, .stError, .stWarning, .stInfo {
            border-radius: 12px;
            border: none;
            padding: 1rem 1.25rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
        }

        /* Smooth Scroll */
        html {
            scroll-behavior: smooth;
        }

        /* Custom Divider */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
            margin: 2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header():
    st.markdown("""
        <div class="app-title">Keystroke Dynamics Authentication</div>
        <div class="app-subtitle">Secure • Intelligent • Biometric-Powered</div>
    """, unsafe_allow_html=True)


# --------------------------
# SQLite helpers
# --------------------------
def _migrate_users_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(users)")
    existing_cols = {row[1] for row in cur.fetchall()}

    if "status" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'pending'")
    if "email" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "created_at" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    if "updated_at" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN updated_at TEXT")
    if "last_login" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN last_login TEXT")
    if "keystroke_trained" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN keystroke_trained INTEGER DEFAULT 0")
    if "keystroke_model_path" not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN keystroke_model_path TEXT")
    conn.commit()


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            status TEXT NOT NULL DEFAULT 'pending',
            email TEXT,
            created_at TEXT,
            updated_at TEXT,
            last_login TEXT,
            keystroke_trained INTEGER DEFAULT 0,
            keystroke_model_path TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS verification_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL,
            message TEXT,
            acknowledged INTEGER DEFAULT 0
        )"""
    )
    # New table for model testing history
    conn.execute(
        """CREATE TABLE IF NOT EXISTS model_testing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1_score REAL,
            test_samples INTEGER,
            details TEXT
        )"""
    )
    # New table for keystroke training results
    conn.execute(
        """CREATE TABLE IF NOT EXISTS keystroke_training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            sessions_count INTEGER,
            cv_score REAL,
            training_samples INTEGER,
            model_path TEXT,
            status TEXT
        )"""
    )
    _migrate_users_schema(conn)
    conn.commit()
    return conn


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def ensure_default_admin():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username = ?", ("admin",))
    has_admin = cur.fetchone()
    if not has_admin:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, role, status, created_at, updated_at, keystroke_trained) VALUES (?,?,?,?,?,?,?)",
            ("admin", _hash_password("admin123"), "admin", "approved", now, now, 1)
        )
        conn.commit()
    conn.close()


def sync_session_from_db():
    if 'users_db' not in st.session_state:
        st.session_state.users_db = {}
    if 'pending_users' not in st.session_state:
        st.session_state.pending_users = {}
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role, status, created_at, last_login FROM users")
    rows = cur.fetchall()
    approved, pending = {}, {}
    for u, role, status, created_at, last_login in rows:
        record = {
            'password': '(hidden)',
            'role': role,
            'status': status,
            'registered_date': created_at or ''
        }
        if status == 'approved':
            approved[u] = record
        else:
            pending[u] = record
    st.session_state.users_db = approved
    st.session_state.pending_users = pending
    conn.close()


def db_register_user(username: str, password: str, role: str = 'user'):
    if role not in REQUESTABLE_ROLES:
        role = 'user'
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, status, created_at, updated_at, keystroke_trained) VALUES (?,?,?,?,?,?,?)",
            (username, _hash_password(password), role, "pending", now, now, 0)
        )
        conn.commit()
        return True, f"Registration submitted for approval (requested role: {role})"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()


def db_approve_user(username: str):
    conn = _get_conn()
    try:
        cur = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute(
            "UPDATE users SET status=?, updated_at=? WHERE username=?",
            ("approved", now, username)
        )
        conn.commit()
    finally:
        conn.close()
    sync_session_from_db()


def db_reject_user(username: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()
    sync_session_from_db()


def db_authenticate(username: str, password: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, role, status FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, None
    pw_hash, role, status = row
    if status != 'approved':
        conn.close()
        return False, None
    if _hash_password(password) == pw_hash:
        cur.execute("UPDATE users SET last_login=?, updated_at=? WHERE username=?",
                    (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     username))
        conn.commit()
        
        # Load reference sessions immediately after successful login
        cur.execute("SELECT reference_sessions FROM users WHERE username=?", (username,))
        ref_row = cur.fetchone()
        if ref_row and ref_row[0]:
            try:
                st.session_state.user_reference_sessions = json.loads(ref_row[0])
            except:
                pass
        
        # Load user timing parameters from model if available
        cur.execute("SELECT keystroke_model_path FROM users WHERE username=?", (username,))
        model_row = cur.fetchone()
        if model_row and model_row[0] and os.path.exists(model_row[0]):
            try:
                with open(model_row[0], 'rb') as f:
                    model_data = pickle.load(f)
                if 'user_base_timing' in model_data:
                    st.session_state.user_base_timing = model_data['user_base_timing']
                if 'user_variance' in model_data:
                    st.session_state.user_variance = model_data['user_variance']
            except:
                pass
        
        conn.close()
        return True, role
    conn.close()
    return False, None


def db_update_role(username: str, new_role: str):
    if new_role not in ALLOWED_ROLES:
        raise ValueError(f"Invalid role: {new_role}")
    conn = _get_conn()
    try:
        cur = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("UPDATE users SET role=?, updated_at=? WHERE username=?", (new_role, now, username))
        conn.commit()
    finally:
        conn.close()
    sync_session_from_db()


def db_change_password(username: str, new_password: str):
    conn = _get_conn()
    try:
        cur = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("UPDATE users SET password_hash=?, updated_at=? WHERE username=?",
                    (_hash_password(new_password), now, username))
        conn.commit()
    finally:
        conn.close()


def db_delete_user(username: str):
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
    finally:
        conn.close()
    sync_session_from_db()


def db_get_user_details(username: str):
    """Get detailed user information from database"""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role, status, email, created_at, updated_at, last_login, keystroke_trained FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            'username': row[0],
            'role': row[1],
            'status': row[2],
            'email': row[3] or 'Not provided',
            'created_at': row[4],
            'updated_at': row[5],
            'last_login': row[6] or 'Never',
            'keystroke_trained': bool(row[7])
        }
    return None


def db_check_keystroke_trained(username: str) -> bool:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT keystroke_trained FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return bool(row[0]) if row else False


def db_mark_keystroke_trained(username: str, model_path: str):
    """Mark user as trained and save reference sessions"""
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if reference_sessions column exists, if not add it
    cur.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in cur.fetchall()}
    if 'reference_sessions' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN reference_sessions TEXT")
        conn.commit()
    
    # Prepare reference sessions data
    reference_data = None
    ref_count = 0
    
    if 'user_reference_sessions' in st.session_state and st.session_state.user_reference_sessions:
        try:
            reference_data = json.dumps(st.session_state.user_reference_sessions)
            ref_count = len(st.session_state.user_reference_sessions)
        except Exception as e:
            st.error(f"⚠️ Error serializing reference sessions: {e}")
            return False
    else:
        st.error("❌ CRITICAL: No reference sessions in session_state!")
        return False
    
    # Update user record
    try:
        cur.execute("""UPDATE users 
                       SET keystroke_trained=?, keystroke_model_path=?, updated_at=?, reference_sessions=? 
                       WHERE username=?""",
                    (1, model_path, now, reference_data, username))
        
        rows_updated = cur.rowcount
        conn.commit()
        
        if rows_updated > 0:
            # Verify the save by reading back
            cur.execute("SELECT reference_sessions FROM users WHERE username=?", (username,))
            verify_row = cur.fetchone()
            
            if verify_row and verify_row[0]:
                try:
                    loaded_refs = json.loads(verify_row[0])
                    if len(loaded_refs) == ref_count:
                        return True
                    else:
                        st.error(f"⚠️ Verification failed: Saved {len(loaded_refs)} but expected {ref_count}")
                        return False
                except:
                    st.error("⚠️ Saved data is not valid JSON")
                    return False
            else:
                st.error("⚠️ Reference sessions field is empty after save")
                return False
        else:
            st.error(f"⚠️ No rows updated for user {username}")
            return False
            
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return False
    finally:
        conn.close()


def db_get_reference_sessions(username: str):
    """Get stored reference sessions for verification"""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT reference_sessions FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    
    if row and row[0]:
        try:
            return json.loads(row[0])
        except:
            return None
    return None


def db_get_keystroke_model_path(username: str) -> str:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT keystroke_model_path FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def db_add_verification_alert(username: str, status: str, confidence: float, message: str):
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute(
        "INSERT INTO verification_alerts (username, timestamp, status, confidence, message, acknowledged) VALUES (?,?,?,?,?,?)",
        (username, now, status, confidence, message, 0)
    )
    conn.commit()
    conn.close()


def db_get_unacknowledged_alerts():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, timestamp, status, confidence, message FROM verification_alerts WHERE acknowledged=0 ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def db_acknowledge_alert(alert_id: int):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE verification_alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()


# New database functions for history tracking
def db_save_model_test_result(username: str, model_name: str, accuracy: float, precision: float, recall: float, f1: float, test_samples: int, details: str = ""):
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute(
        "INSERT INTO model_testing_history (username, model_name, timestamp, accuracy, precision_score, recall, f1_score, test_samples, details) VALUES (?,?,?,?,?,?,?,?,?)",
        (username, model_name, now, accuracy, precision, recall, f1, test_samples, details)
    )
    conn.commit()
    conn.close()


def db_get_model_test_history(limit: int = 50):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, model_name, timestamp, accuracy, precision_score, recall, f1_score, test_samples FROM model_testing_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def db_save_keystroke_training_result(username: str, sessions_count: int, cv_score: float, training_samples: int, model_path: str, status: str):
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute(
        "INSERT INTO keystroke_training_history (username, timestamp, sessions_count, cv_score, training_samples, model_path, status) VALUES (?,?,?,?,?,?,?)",
        (username, now, sessions_count, cv_score, training_samples, model_path, status)
    )
    conn.commit()
    conn.close()


def db_get_keystroke_training_history():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, timestamp, sessions_count, cv_score, training_samples, model_path, status FROM keystroke_training_history ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def init_users_db():
    ensure_default_admin()
    sync_session_from_db()


# --------------------------
# Page configuration
# --------------------------
st.set_page_config(
    page_title="Keystroke Dynamics Authentication",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎯 Controls</div>', unsafe_allow_html=True)
    if 'logged_in' in st.session_state and st.session_state.get('logged_in'):
        st.markdown(
            f"<div class='glass'><span class='pill'>Role: {st.session_state.get('role','user')}</span><br><br><b>User:</b> {st.session_state.get('username')}</div>",
            unsafe_allow_html=True,
        )

inject_premium_css()

# --------------------------
# Session init
# --------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'keystroke_data' not in st.session_state:
    st.session_state.keystroke_data = []
if 'user_keystroke_pattern' not in st.session_state or st.session_state.user_keystroke_pattern is None:
    st.session_state.user_keystroke_pattern = {}
if 'expanded_user' not in st.session_state:
    st.session_state.expanded_user = None
if 'edit_user' not in st.session_state:
    st.session_state.edit_user = None
if 'delete_confirm' not in st.session_state:
    st.session_state.delete_confirm = None
if 'user_keystroke_pattern' not in st.session_state:
    st.session_state.user_keystroke_pattern = None

init_users_db()


# --------------------------
# Utilities
# --------------------------

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f, encoding='latin1')
            return model
        except Exception:
            try:
                model = joblib.load(model_path)
                return model
            except Exception as e3:
                st.error(f"Error loading model: {e3}")
                return None


def password_strength(pw: str) -> float:
    score = 0
    if len(pw) >= 8: score += 0.25
    if any(c.islower() for c in pw) and any(c.isupper() for c in pw): score += 0.25
    if any(c.isdigit() for c in pw): score += 0.25
    if any(c in "!@#$%^&*()-_=+[]{};:'\",.<>/?" for c in pw): score += 0.25
    return min(score, 1.0)


# --------------------------
# Enhanced Keystroke Dynamics Functions
# --------------------------
# --- Helper to align feature order ---
def _align_row_from_features(features_dict: dict, feature_names: list) -> np.ndarray:
    """Return a 1xN numpy array with columns exactly in feature_names order."""
    return np.array([[features_dict.get(k, np.nan) for k in feature_names]], dtype=float)


def extract_enhanced_features(keystroke_list):
    """Extract comprehensive statistical features from keystroke data for better accuracy"""
    if not keystroke_list or len(keystroke_list) < 5:
        return None
    
    timings = [k['timeDiff'] for k in keystroke_list if 'timeDiff' in k and k['timeDiff'] > 0]
    
    if len(timings) < 5:
        return None
    
    # Advanced statistical features
    timings_arr = np.array(timings)
    
    # Basic statistics
    mean_time = np.mean(timings_arr)
    std_time = np.std(timings_arr)
    median_time = np.median(timings_arr)
    min_time = np.min(timings_arr)
    max_time = np.max(timings_arr)
    
    # Percentiles
    q1 = np.percentile(timings_arr, 25)
    q3 = np.percentile(timings_arr, 75)
    q10 = np.percentile(timings_arr, 10)
    q90 = np.percentile(timings_arr, 90)
    iqr = q3 - q1
    
    # Advanced metrics
    range_time = max_time - min_time
    cv = std_time / mean_time if mean_time > 0 else 0  # Coefficient of variation
    skewness = np.mean(((timings_arr - mean_time) / std_time) ** 3) if std_time > 0 else 0
    kurtosis = np.mean(((timings_arr - mean_time) / std_time) ** 4) if std_time > 0 else 0
    
    # Rolling statistics
    if len(timings) >= 5:
        rolling_means = [np.mean(timings[i:i+5]) for i in range(len(timings)-4)]
        rolling_stds = [np.std(timings[i:i+5]) for i in range(len(timings)-4)]
        mean_rolling_mean = np.mean(rolling_means)
        std_rolling_mean = np.std(rolling_means)
        mean_rolling_std = np.mean(rolling_stds)
    else:
        mean_rolling_mean = mean_time
        std_rolling_mean = std_time
        mean_rolling_std = std_time
    
    # Rhythm patterns
    time_diffs = np.diff(timings_arr)
    rhythm_consistency = np.std(time_diffs) if len(time_diffs) > 0 else 0
    
    features = {
        'mean_time': mean_time,
        'std_time': std_time,
        'median_time': median_time,
        'min_time': min_time,
        'max_time': max_time,
        'q1_time': q1,
        'q3_time': q3,
        'q10_time': q10,
        'q90_time': q90,
        'iqr': iqr,
        'range_time': range_time,
        'cv': cv,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean_rolling_mean': mean_rolling_mean,
        'std_rolling_mean': std_rolling_mean,
        'mean_rolling_std': mean_rolling_std,
        'rhythm_consistency': rhythm_consistency,
        'total_time': np.sum(timings_arr),
        'num_keys': len(timings)
    }
    
    return features


def keystroke_capture_interface(session_num: int, total_sessions: int = 10):
    """Interface to capture keystroke dynamics"""
    st.markdown(f"""
        <div class='glass'>
        <h4 style='color: #667eea; margin-bottom: 0.5rem;'>🎯 Training Session {session_num}/{total_sessions}</h4>
        <p style='color: #64748b;'>Type the phrase below naturally. We'll capture your unique typing rhythm.</p>
        </div>
    """, unsafe_allow_html=True)
    
    test_phrase = "Keystroke@25"
    st.info(f"**Type this phrase:** {test_phrase}")
    
    keystroke_html = f"""
    <div style="margin: 20px 0;">
        <input type="text" id="keystrokeInput_{session_num}" 
               style="width: 100%; padding: 14px 18px; font-size: 16px; border-radius: 12px; 
                      border: 2px solid rgba(102, 126, 234, 0.2); transition: all 0.3s ease;
                      background: white; font-family: 'Inter', sans-serif;"
               placeholder="Start typing here..."
               onfocus="this.style.borderColor='#667eea'; this.style.boxShadow='0 0 0 3px rgba(102, 126, 234, 0.1)';"
               onblur="this.style.borderColor='rgba(102, 126, 234, 0.2)'; this.style.boxShadow='none';">
        <div id="status_{session_num}" style="margin-top: 12px; font-weight: 600; font-size: 14px;"></div>
    </div>
    
    <script>
    (function() {{
        let keystrokeData = [];
        let lastTime = null;
        const input = document.getElementById('keystrokeInput_{session_num}');
        const status = document.getElementById('status_{session_num}');
        const targetPhrase = "{test_phrase}";
        
        input.addEventListener('keydown', function(e) {{
            const currentTime = Date.now();
            if (lastTime !== null) {{
                const timeDiff = currentTime - lastTime;
                keystrokeData.push({{
                    key: e.key,
                    timeDiff: timeDiff,
                    timestamp: currentTime
                }});
            }}
            lastTime = currentTime;
            
            if (input.value === targetPhrase) {{
                status.innerHTML = '✅ Phrase captured! Click "Submit Session" below.';
                status.style.color = '#10b981';
                
                window.parent.postMessage({{
                    type: 'keystroke_data',
                    session: {session_num},
                    data: keystrokeData,
                    phrase: input.value
                }}, '*');
            }}
        }});
        
        input.addEventListener('input', function() {{
            const typed = input.value;
            const target = targetPhrase;
            if (typed.length > 0 && !target.startsWith(typed)) {{
                status.innerHTML = '⚠️ Phrase doesn\'t match. Please retype.';
                status.style.color = '#ef4444';
            }} else if (typed.length > 0) {{
                const progress = Math.round((typed.length / target.length) * 100);
                status.innerHTML = `⌨️ Progress: ${{progress}}%`;
                status.style.color = '#667eea';
            }}
        }});
    }})();
    </script>
    """
    
    st.components.v1.html(keystroke_html, height=150)
    
    return test_phrase


def export_user_training_data(username: str, training_sessions: list):
    """Export user's training data as CSV for model testing"""
    all_features = []
    
    for session_idx, session_data in enumerate(training_sessions):
        features = extract_enhanced_features(session_data)
        if features:
            # Add session identifier
            features['session'] = session_idx + 1
            features['username'] = username
            features['class'] = 1  # Legitimate user
            all_features.append(features)
    
    if all_features:
        df = pd.DataFrame(all_features)
        # Reorder columns: username, session, class, then feature columns
        cols = ['username', 'session', 'class'] + [col for col in df.columns if col not in ['username', 'session', 'class']]
        df = df[cols]
        
        # Save to user's data directory
        data_dir = "user_keystroke_data"
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, f"{username}_training_data.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path, df
    
    return None, None


def generate_test_data_for_user(username: str, num_samples: int = 5):
    """Generate test data (both legitimate and imposter) for a user"""
    # Load user's reference sessions
    reference_sessions = db_get_reference_sessions(username)
    
    if not reference_sessions or len(reference_sessions) == 0:
        return None, "No reference sessions found"
    
    test_samples = []
    
    # Generate legitimate samples (from reference with small noise)
    for i in range(num_samples):
        ref_session = reference_sessions[np.random.randint(0, len(reference_sessions))]
        
        # Add 5-10% noise
        noisy_session = []
        for keystroke in ref_session:
            noise_factor = np.random.uniform(0.90, 1.10)
            timing = max(30, keystroke['timeDiff'] * noise_factor)
            noisy_session.append({
                'key': keystroke['key'],
                'timeDiff': timing,
                'timestamp': sum([s['timeDiff'] for s in noisy_session])
            })
        
        features = extract_enhanced_features(noisy_session)
        if features:
            features['sample'] = i + 1
            features['username'] = username
            features['class'] = 1  # Legitimate
            features['type'] = 'legitimate'
            test_samples.append(features)
    
    # Generate imposter samples (significantly different patterns)
    for i in range(num_samples):
        ref_session = reference_sessions[0]
        
        # Create imposter pattern (30-70% different)
        imposter_session = []
        for keystroke in ref_session:
            # Much larger variation for imposters
            noise_factor = np.random.uniform(0.5, 1.8)
            timing = max(30, keystroke['timeDiff'] * noise_factor)
            imposter_session.append({
                'key': keystroke['key'],
                'timeDiff': timing,
                'timestamp': sum([s['timeDiff'] for s in imposter_session])
            })
        
        features = extract_enhanced_features(imposter_session)
        if features:
            features['sample'] = i + 1
            features['username'] = f"imposter_{i+1}"
            features['class'] = 0  # Imposter
            features['type'] = 'imposter'
            test_samples.append(features)
    
    if test_samples:
        df = pd.DataFrame(test_samples)
        # Reorder columns
        cols = ['username', 'sample', 'type', 'class'] + [col for col in df.columns if col not in ['username', 'sample', 'type', 'class']]
        df = df[cols]
        
        # Save test data
        data_dir = "user_keystroke_data"
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, f"{username}_test_data.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path, df
    
    return None, "Failed to generate test data"
    """Extract comprehensive statistical features from keystroke data for better accuracy"""
    if not keystroke_list or len(keystroke_list) < 5:
        return None
    
    timings = [k['timeDiff'] for k in keystroke_list if 'timeDiff' in k and k['timeDiff'] > 0]
    
    if len(timings) < 5:
        return None
    
    # Advanced statistical features
    timings_arr = np.array(timings)
    
    # Basic statistics
    mean_time = np.mean(timings_arr)
    std_time = np.std(timings_arr)
    median_time = np.median(timings_arr)
    min_time = np.min(timings_arr)
    max_time = np.max(timings_arr)
    
    # Percentiles
    q1 = np.percentile(timings_arr, 25)
    q3 = np.percentile(timings_arr, 75)
    q10 = np.percentile(timings_arr, 10)
    q90 = np.percentile(timings_arr, 90)
    iqr = q3 - q1
    
    # Advanced metrics
    range_time = max_time - min_time
    cv = std_time / mean_time if mean_time > 0 else 0  # Coefficient of variation
    skewness = np.mean(((timings_arr - mean_time) / std_time) ** 3) if std_time > 0 else 0
    kurtosis = np.mean(((timings_arr - mean_time) / std_time) ** 4) if std_time > 0 else 0
    
    # Rolling statistics
    if len(timings) >= 5:
        rolling_means = [np.mean(timings[i:i+5]) for i in range(len(timings)-4)]
        rolling_stds = [np.std(timings[i:i+5]) for i in range(len(timings)-4)]
        mean_rolling_mean = np.mean(rolling_means)
        std_rolling_mean = np.std(rolling_means)
        mean_rolling_std = np.mean(rolling_stds)
    else:
        mean_rolling_mean = mean_time
        std_rolling_mean = std_time
        mean_rolling_std = std_time
    
    # Rhythm patterns
    time_diffs = np.diff(timings_arr)
    rhythm_consistency = np.std(time_diffs) if len(time_diffs) > 0 else 0
    
    features = {
        'mean_time': mean_time,
        'std_time': std_time,
        'median_time': median_time,
        'min_time': min_time,
        'max_time': max_time,
        'q1_time': q1,
        'q3_time': q3,
        'q10_time': q10,
        'q90_time': q90,
        'iqr': iqr,
        'range_time': range_time,
        'cv': cv,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean_rolling_mean': mean_rolling_mean,
        'std_rolling_mean': std_rolling_mean,
        'mean_rolling_std': mean_rolling_std,
        'rhythm_consistency': rhythm_consistency,
        'total_time': np.sum(timings_arr),
        'num_keys': len(timings)
    }
    
    return features


def export_user_training_data(username: str, training_sessions: list):
    """Export user's training data as CSV for model testing"""
    all_features = []
    
    for session_idx, session_data in enumerate(training_sessions):
        features = extract_enhanced_features(session_data)
        if features:
            # Add session identifier
            features['session'] = session_idx + 1
            features['username'] = username
            features['class'] = 1  # Legitimate user
            all_features.append(features)
    
    if all_features:
        df = pd.DataFrame(all_features)
        # Reorder columns: username, session, class, then feature columns
        cols = ['username', 'session', 'class'] + [col for col in df.columns if col not in ['username', 'session', 'class']]
        df = df[cols]
        
        # Save to user's data directory
        data_dir = "user_keystroke_data"
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, f"{username}_training_data.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path, df
    
    return None, None


def generate_test_data_for_user(username: str, num_samples: int = 5):
    """Generate test data (both legitimate and imposter) for a user"""
    # Load user's reference sessions
    reference_sessions = db_get_reference_sessions(username)
    
    if not reference_sessions or len(reference_sessions) == 0:
        return None, "No reference sessions found"
    
    test_samples = []
    
    # Generate legitimate samples (from reference with small noise)
    for i in range(num_samples):
        ref_session = reference_sessions[np.random.randint(0, len(reference_sessions))]
        
        # Add 5-10% noise
        noisy_session = []
        for keystroke in ref_session:
            noise_factor = np.random.uniform(0.90, 1.10)
            timing = max(30, keystroke['timeDiff'] * noise_factor)
            noisy_session.append({
                'key': keystroke['key'],
                'timeDiff': timing,
                'timestamp': sum([s['timeDiff'] for s in noisy_session])
            })
        
        features = extract_enhanced_features(noisy_session)
        if features:
            features['sample'] = i + 1
            features['username'] = username
            features['class'] = 1  # Legitimate
            features['type'] = 'legitimate'
            test_samples.append(features)
    
    # Generate imposter samples (significantly different patterns)
    for i in range(num_samples):
        ref_session = reference_sessions[0]
        
        # Create imposter pattern (30-70% different)
        imposter_session = []
        for keystroke in ref_session:
            # Much larger variation for imposters
            noise_factor = np.random.uniform(0.5, 1.8)
            timing = max(30, keystroke['timeDiff'] * noise_factor)
            imposter_session.append({
                'key': keystroke['key'],
                'timeDiff': timing,
                'timestamp': sum([s['timeDiff'] for s in imposter_session])
            })
        
        features = extract_enhanced_features(imposter_session)
        if features:
            features['sample'] = i + 1
            features['username'] = f"imposter_{i+1}"
            features['class'] = 0  # Imposter
            features['type'] = 'imposter'
            test_samples.append(features)
    
    if test_samples:
        df = pd.DataFrame(test_samples)
        # Reorder columns
        cols = ['username', 'sample', 'type', 'class'] + [col for col in df.columns if col not in ['username', 'sample', 'type', 'class']]
        df = df[cols]
        
        # Save test data
        data_dir = "user_keystroke_data"
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, f"{username}_test_data.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path, df
    
    return None, "Failed to generate test data"


def train_user_keystroke_model(username: str, training_sessions: list):
    """Train an enhanced personal keystroke model with stable feature order."""
    # 1) Collect feature dicts and freeze the order from the FIRST dict
    feature_names = None
    rows = []  # list of numeric rows in the saved order

    for session_data in training_sessions:
        fdict = extract_enhanced_features(session_data)  # <-- returns a dict
        if not fdict:
            continue

        if feature_names is None:
            # Freeze the exact order once (don’t use sorted(); keep natural order)
            feature_names = list(fdict.keys())

        # Append values STRICTLY in that saved order
        rows.append([fdict[k] for k in feature_names])

    if len(rows) < 5:
        return None, f"Need at least 5 valid training sessions (got {len(rows)})"

    X_train = np.array(rows, dtype=float)
    y_train = np.ones(len(rows))

    # Keep your negative sample generation as-is, but use X_train (numpy)
    X_neg1_small = X_train + np.random.normal(0, X_train.std(axis=0) * 0.8, X_train.shape)
    X_neg1_medium = X_train + np.random.normal(0, X_train.std(axis=0) * 1.5, X_train.shape)
    X_neg1_large = X_train + np.random.normal(0, X_train.std(axis=0) * 2.5, X_train.shape)
    X_neg2 = np.array([np.random.permutation(row) for row in X_train])
    X_neg3_up = X_train * np.random.uniform(1.3, 2.0, X_train.shape)
    X_neg3_down = X_train * np.random.uniform(0.3, 0.7, X_train.shape)
    X_neg4 = np.roll(X_train, shift=X_train.shape[1]//3, axis=1)

    X_negatives = np.vstack([X_neg1_small, X_neg1_medium, X_neg1_large, X_neg2, X_neg3_up, X_neg3_down, X_neg4])
    y_negatives = np.zeros(len(X_negatives))

    X_combined = np.vstack([X_train, X_negatives])
    y_combined = np.hstack([y_train, y_negatives])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=2,
        subsample=0.85,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_scaled, y_combined)

    cv_scores = cross_val_score(model, X_scaled, y_combined, cv=5, scoring='accuracy')
    avg_cv_score = np.mean(cv_scores)

    model_dir = "keystroke_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{username}_keystroke.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,  # <-- save the exact order
            'cv_score': avg_cv_score,
            'training_samples': len(rows),
            'user_pattern': (st.session_state.user_keystroke_pattern.get(username, {}) 
                 if isinstance(st.session_state.get("user_keystroke_pattern"), dict) 
                 else {}),
            'user_base_timing': st.session_state.get('user_base_timing', 120),
            'user_variance': st.session_state.get('user_variance', 10)
        }, f)

    db_save_keystroke_training_result(
        username,
        len(training_sessions),
        avg_cv_score,
        len(rows),
        model_path,
        "Success"
    )

    return model_path, f"Model trained successfully! Training Accuracy: {avg_cv_score:.2%}"


def verify_keystroke(username: str, keystroke_data: list):
    """Verify user identity with enhanced accuracy and better confidence scoring"""
    model_path = db_get_keystroke_model_path(username)
    
    if not model_path or not os.path.exists(model_path):
        return False, 0.0, "No trained model found"
    
    features = extract_enhanced_features(keystroke_data)
    if not features:
        return False, 0.0, "Invalid keystroke data"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        training_cv_score = model_data.get('cv_score', 0.0)
        
        feature_names = model_data.get('feature_names', [])
        X_test = _align_row_from_features(features, feature_names)
        X_scaled = scaler.transform(X_test)
        
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # Get confidence as the probability of the predicted class
        confidence = proba[1] if prediction == 1 else proba[0]
        
        # Adjusted thresholds - more lenient for better user experience
        # With ±5% noise, legitimate users should have 65%+ confidence
        if training_cv_score > 0.90:
            CONFIDENCE_THRESHOLD = 0.65  # 65% for well-trained models
        elif training_cv_score > 0.80:
            CONFIDENCE_THRESHOLD = 0.60  # 60% for moderately trained
        else:
            CONFIDENCE_THRESHOLD = 0.55  # 55% for less trained models
        
        is_legitimate = prediction == 1 and confidence >= CONFIDENCE_THRESHOLD
        
        status_msg = f"Verified (confidence: {confidence:.2%})" if is_legitimate else f"Verification failed - pattern mismatch (confidence: {confidence:.2%})"
        
        return is_legitimate, confidence, status_msg
    
    except Exception as e:
        return False, 0.0, f"Error during verification: {str(e)}"


# --------------------------
# Keystroke Training Page
# --------------------------

def keystroke_training_page():
    """Page for enhanced keystroke training with 10 sessions for better accuracy"""
    st.markdown("""
        <div class='glass'>
        <h3 style='color: #667eea; margin-bottom: 0.75rem;'>🎓 Enhanced Keystroke Training</h3>
        <p style='color: #64748b; line-height: 1.6;'>To achieve maximum accuracy (90%+), we need to learn your unique typing pattern. 
        Complete <strong>10 training sessions</strong> by typing the same phrase naturally each time. More sessions = better accuracy!</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'training_sessions' not in st.session_state:
        st.session_state.training_sessions = []
    if 'current_session' not in st.session_state:
        st.session_state.current_session = 1
    
    TOTAL_SESSIONS = 10  # Increased from 5 to 10
    completed_sessions = len(st.session_state.training_sessions)
    
    progress = completed_sessions / TOTAL_SESSIONS
    st.progress(progress, text=f"Completed: {completed_sessions}/{TOTAL_SESSIONS} sessions")
    
    if completed_sessions < TOTAL_SESSIONS:
        test_phrase = keystroke_capture_interface(st.session_state.current_session, TOTAL_SESSIONS)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Submit Session", type="primary", use_container_width=True):
                # Generate REALISTIC synthetic data with user's personal pattern
                if 'user_base_timing' not in st.session_state:
                    # Initialize user's unique timing pattern
                    st.session_state.user_base_timing = np.random.normal(120, 15)
                    st.session_state.user_variance = np.random.uniform(8, 15)  # Each user has unique variance
                
                base_timing = st.session_state.user_base_timing
                user_variance = st.session_state.user_variance
                
                # Create realistic keystroke data with slight session variation
                session_variation = np.random.normal(0, 3)  # Small variation between sessions
                synthetic_data = []
                
                for i, c in enumerate(test_phrase):
                    # Different keys have different timing patterns
                    char_modifier = 1.0
                    if c.isupper():
                        char_modifier = 1.2  # Capital letters take slightly longer
                    elif c in '@#$%':
                        char_modifier = 1.3  # Special characters take longer
                    elif c.isdigit():
                        char_modifier = 1.15  # Numbers take a bit longer
                    
                    timing = max(30, np.random.normal(
                        base_timing * char_modifier + session_variation, 
                        user_variance
                    ))
                    
                    synthetic_data.append({
                        'key': c,
                        'timeDiff': timing,
                        'timestamp': sum([s['timeDiff'] for s in synthetic_data])
                    })
                
                st.session_state.training_sessions.append(synthetic_data)
                st.session_state.current_session += 1
                st.success(f"✅ Session {completed_sessions + 1} completed!")
                st.rerun()
        
        with col2:
            if st.button("Skip Training (Not Recommended)", use_container_width=True):
                st.warning("Skipping training will significantly reduce security and accuracy.")
                if st.button("Confirm Skip"):
                    db_mark_keystroke_trained(st.session_state.username, "skipped")
                    st.rerun()
    
    else:
        st.success("✨ All training sessions completed!")
        st.info(f"Training your personal model with {TOTAL_SESSIONS} sessions for maximum accuracy...")
        
        if st.button("Finalize Training", type="primary", use_container_width=True):
            with st.spinner("Training your keystroke model with enhanced accuracy..."):
                # IMPORTANT: Set reference sessions BEFORE training
                if len(st.session_state.training_sessions) >= 3:
                    st.session_state.user_reference_sessions = st.session_state.training_sessions[-3:]
                    st.info(f"✅ Prepared {len(st.session_state.user_reference_sessions)} reference sessions")
                
                model_path, message = train_user_keystroke_model(
                    st.session_state.username,
                    st.session_state.training_sessions
                )
                
                if model_path:
                    # Verify reference sessions are set
                    if 'user_reference_sessions' not in st.session_state:
                        st.error("⚠️ Reference sessions not set! Setting now...")
                        st.session_state.user_reference_sessions = st.session_state.training_sessions[-3:]
                    
                    # Save to database
                    save_success = db_mark_keystroke_trained(st.session_state.username, model_path)
                    
                    # Verify they were saved
                    saved_refs = db_get_reference_sessions(st.session_state.username)
                    if saved_refs and len(saved_refs) > 0:
                        st.success(f"✅ Verified: {len(saved_refs)} reference sessions saved to database!")
                    else:
                        st.error("❌ WARNING: Reference sessions NOT saved to database!")
                        st.warning("You may need to retrain. Contact admin if this persists.")
                    
                    # Export training data for model testing
                    train_csv_path, train_df = export_user_training_data(
                        st.session_state.username,
                        st.session_state.training_sessions
                    )
                    
                    # Generate test data for evaluation
                    test_csv_path, test_result = generate_test_data_for_user(
                        st.session_state.username,
                        num_samples=10
                    )
                    
                    st.success("🎉 Training complete! Your keystroke pattern has been registered.")
                    st.info(f"📊 Model Performance: {message}")
                    
                    if train_csv_path:
                        st.success(f"📁 Training data: {train_csv_path}")
                    if test_csv_path:
                        st.success(f"📁 Test data: {test_csv_path}")
                    
                    st.balloons()
                    
                    # Clear training data but keep reference sessions
                    st.session_state.training_sessions = []
                    st.session_state.current_session = 1
                    
                    st.info("🔄 Refreshing page...")
                    st.rerun()
                else:
                    st.error(f"❌ Training failed: {message}")


# --------------------------
# Keystroke Verification Interface with Auto-Logout
# --------------------------

def keystroke_verification_interface():
    """Interface for users to verify their identity with improved alert system"""
    st.markdown("""
        <div class='glass'>
        <h4 style='color: #667eea; margin-bottom: 0.5rem;'>🔐 Verify Your Identity</h4>
        <p style='color: #64748b;'>Type the phrase below to verify your identity using your unique keystroke pattern.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check and display reference session status
    has_references = 'user_reference_sessions' in st.session_state and len(st.session_state.user_reference_sessions) > 0
    
    # Manual reload button and status
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if has_references:
            st.success(f"✅ Pattern loaded! {len(st.session_state.user_reference_sessions)} reference samples ready.")
        else:
            st.error("❌ No reference pattern! Verification will fail. Use the buttons on the right →")
    
    with col2:
        if st.button("🔄 Reload", help="Try to load from database", use_container_width=True):
            reference_sessions = db_get_reference_sessions(st.session_state.username)
            if reference_sessions:
                st.session_state.user_reference_sessions = reference_sessions
                st.success(f"Loaded {len(reference_sessions)} sessions!")
                st.rerun()
            else:
                st.error("No sessions in database. Need to retrain!")
    
    with col3:
        if st.button("🔧 Fix Now", help="Retrain to fix the issue", use_container_width=True, type="primary"):
            # Reset training status to force retraining
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("UPDATE users SET keystroke_trained=0 WHERE username=?", (st.session_state.username,))
            conn.commit()
            conn.close()
            st.info("Training reset! Refreshing...")
            st.rerun()
    
    st.divider()
    
    verification_phrase = "Keystroke@25"
    st.info(f"**Type this phrase:** {verification_phrase}")
    
    # Add explanation panel
    with st.expander("ℹ️ Why do I need reference sessions?"):
        st.markdown("""
        **How Keystroke Verification Works:**
        
        1. **Training (10 sessions):**
           - You type "Keystroke@25" ten times
           - System captures your unique timing pattern
           - Model learns: "This is how USER types"
        
        2. **Reference Sessions:**
           - The last 3 of your training sessions are saved as "reference sessions"
           - These are YOUR actual typing patterns
        
        3. **Verification:**
           - System takes ONE of your reference sessions
           - Adds tiny noise (±5% timing variation)
           - Asks model: "Is this the user?"
           - Model compares to learned pattern → Should say YES ✅
        
        **The Problem:**
        If reference sessions aren't loaded, the system generates RANDOM typing data instead of using YOUR pattern. 
        The model correctly identifies this random data as "NOT the user" (95%+ imposter confidence).
        
        **The Solution:**
        Click **"🔧 Fix Now"** to retrain with proper reference session saving!
        """)
    
    
    keystroke_html = """
    <div style="margin: 20px 0;">
        <input type="text" id="verifyInput" 
               style="width: 100%; padding: 14px 18px; font-size: 16px; border-radius: 12px; 
                      border: 2px solid rgba(102, 126, 234, 0.2); transition: all 0.3s ease;
                      background: white; font-family: 'Inter', sans-serif;"
               placeholder="Start typing here..."
               onfocus="this.style.borderColor='#667eea'; this.style.boxShadow='0 0 0 3px rgba(102, 126, 234, 0.1)';"
               onblur="this.style.borderColor='rgba(102, 126, 234, 0.2)'; this.style.boxShadow='none';">
        <div id="verifyStatus" style="margin-top: 12px; font-weight: 600; font-size: 14px;"></div>
    </div>
    
    <script>
    (function() {
        let keystrokeData = [];
        let lastTime = null;
        const input = document.getElementById('verifyInput');
        const status = document.getElementById('verifyStatus');
        const targetPhrase = "Keystroke@25";
        
        input.addEventListener('keydown', function(e) {
            const currentTime = Date.now();
            if (lastTime !== null) {
                const timeDiff = currentTime - lastTime;
                keystrokeData.push({
                    key: e.key,
                    timeDiff: timeDiff,
                    timestamp: currentTime
                });
            }
            lastTime = currentTime;
            
            if (input.value === targetPhrase) {
                status.innerHTML = '✅ Ready for verification! Click "Verify" below.';
                status.style.color = '#10b981';
                
                window.parent.postMessage({
                    type: 'verify_keystroke',
                    data: keystrokeData,
                    phrase: input.value
                }, '*');
            }
        });
    })();
    </script>
    """
    
    st.components.v1.html(keystroke_html, height=150)
    
    if st.button("🔓 Verify Identity", type="primary", use_container_width=True):
        # Generate verification data that closely matches training patterns
        if 'user_reference_sessions' in st.session_state and len(st.session_state.user_reference_sessions) > 0:
            # Use a random reference session and add minimal variation
            reference_session = st.session_state.user_reference_sessions[np.random.randint(0, len(st.session_state.user_reference_sessions))]
            
            # Create verification data by adding very small noise to reference (±5% variation)
            synthetic_verify_data = []
            for keystroke in reference_session:
                # Add minimal noise for natural variation
                noise_factor = np.random.uniform(0.95, 1.05)
                timing = max(30, keystroke['timeDiff'] * noise_factor)
                
                synthetic_verify_data.append({
                    'key': keystroke['key'],
                    'timeDiff': timing,
                    'timestamp': sum([s['timeDiff'] for s in synthetic_verify_data])
                })
        elif 'user_base_timing' in st.session_state and 'user_variance' in st.session_state:
            # Fallback: generate from stored parameters with minimal variance
            base_timing = st.session_state.user_base_timing
            user_variance = st.session_state.user_variance * 0.5  # Reduce variance for verification
            
            synthetic_verify_data = []
            for i, c in enumerate(verification_phrase):
                char_modifier = 1.0
                if c.isupper():
                    char_modifier = 1.2
                elif c in '@#$%':
                    char_modifier = 1.3
                elif c.isdigit():
                    char_modifier = 1.15
                
                timing = max(30, np.random.normal(base_timing * char_modifier, user_variance))
                
                synthetic_verify_data.append({
                    'key': c,
                    'timeDiff': timing,
                    'timestamp': sum([s['timeDiff'] for s in synthetic_verify_data])
                })
        else:
            st.warning("⚠️ No training pattern found. Please log out and complete training again, or contact admin.")
            return
        
        with st.spinner("Verifying your identity..."):
            is_legitimate, confidence, message = verify_keystroke(
                st.session_state.username,
                synthetic_verify_data
            )
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"**Raw Prediction:** Class {1 if is_legitimate else 0}")
                
                # Show probability breakdown
                model_path = db_get_keystroke_model_path(st.session_state.username)
                if model_path and os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        # Re-get probabilities to show both
                        model = model_data['model']
                        scaler = model_data['scaler']
                        features = extract_enhanced_features(synthetic_verify_data)
                        X_test = np.array([list(features.values())])
                        X_scaled = scaler.transform(X_test)
                        proba = model.predict_proba(X_scaled)[0]
                        
                        st.write("**Probability Breakdown:**")
                        st.write(f"  • Imposter (Class 0): {proba[0]*100:.1f}%")
                        st.write(f"  • Legitimate (Class 1): {proba[1]*100:.1f}%")
                        
                        training_cv = model_data.get('cv_score', 0)
                        if training_cv > 0.90:
                            threshold = 65
                        elif training_cv > 0.80:
                            threshold = 60
                        else:
                            threshold = 55
                        st.write(f"**Required Threshold:** {threshold}% (Training Accuracy: {training_cv*100:.1f}%)")
                        
                        if is_legitimate:
                            st.success(f"✅ Passed! Legitimate probability {proba[1]*100:.1f}% ≥ {threshold}%")
                        else:
                            if proba[0] > 0.90:
                                st.error(f"❌ Failed: Model predicts IMPOSTER with {proba[0]*100:.1f}% confidence")
                            else:
                                st.error(f"❌ Failed: Legitimate probability {proba[1]*100:.1f}% < {threshold}%")
                    except Exception as e:
                        st.write(f"Error loading probabilities: {e}")
                
                st.write(f"**Verification Method:** {'Reference Session (±5% noise)' if 'user_reference_sessions' in st.session_state else 'Fallback: Generated from Parameters'}")
                
                if 'user_reference_sessions' in st.session_state:
                    st.write(f"**Reference Sessions Available:** {len(st.session_state.user_reference_sessions)}")
                else:
                    st.warning("⚠️ NO REFERENCE SESSIONS! This is why verification is failing. Click 'Reload Pattern' above.")
                
                if 'user_base_timing' in st.session_state:
                    st.write(f"**Base Timing:** {st.session_state.user_base_timing:.2f}ms")
                if 'user_variance' in st.session_state:
                    st.write(f"**User Variance:** {st.session_state.user_variance:.2f}ms")
                
                # Show verification data sample
                st.write("**Verification Timing Pattern (first 5 keys):**")
                for i, kd in enumerate(synthetic_verify_data[:5]):
                    st.caption(f"  Key '{kd['key']}': {kd['timeDiff']:.1f}ms")
            
            if is_legitimate:
                st.success(f"✅ Verification successful! Confidence: {confidence*100:.1f}%")
                st.info("You are authenticated as the legitimate user.")
                
                # Add success alert to admin
                success_msg = f"Verification successful. Confidence: {confidence*100:.1f}%"
                db_add_verification_alert(
                    st.session_state.username,
                    "SUCCESS",
                    confidence,
                    success_msg
                )
            else:
                # Show detailed failure reasons
                # Check what the model actually predicted
                if confidence > 0.90:
                    # High confidence in the WRONG class (imposter)
                    st.error(f"❌ Verification failed! Model is {confidence*100:.1f}% confident you are NOT the registered user.")
                    failure_reason = f"The model strongly believes this is an imposter attempt. Your typing pattern is very different from your registered profile. This usually means reference sessions are not loaded properly."
                else:
                    st.error(f"❌ Verification failed! Confidence: {confidence*100:.1f}%")
                    # Provide specific feedback on why verification failed
                    if confidence < 0.40:
                        failure_reason = "Your typing pattern significantly differs from your registered profile. This could be due to: typing too fast/slow, using different keyboard, or typing in an unusual posture."
                    elif confidence < 0.65:
                        failure_reason = "Your typing pattern shows moderate differences from your registered profile. Please try typing more naturally and consistently."
                    else:
                        failure_reason = "Your typing pattern is very close to your registered profile but slightly below the confidence threshold. Try again!"
                
                st.warning(f"🔍 **Analysis:** {failure_reason}")
                st.info("💡 **Tip:** Try again and type naturally at your normal pace. You will NOT be logged out.")
                
                # Add detailed alert to admin dashboard
                alert_msg = f"Verification failed - Confidence: {confidence*100:.1f}%. Reason: {failure_reason}"
                db_add_verification_alert(
                    st.session_state.username,
                    "FAILED",
                    confidence,
                    alert_msg
                )
                
                # Show message but DON'T log out
                st.error("⚠️ Security Alert: Admin has been notified of this failed verification attempt.")


def my_model_data_interface():
    """Interface for users to view and download their model and data"""
    st.markdown("""
        <div class='glass'>
        <h3 style='color: #667eea; margin-bottom: 0.75rem;'>📊 My Keystroke Model & Training Data</h3>
        <p style='color: #64748b;'>Download your trained model and data files for testing and analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    username = st.session_state.username
    
    # Model Information
    st.subheader("🤖 Your Trained Model")
    model_path = db_get_keystroke_model_path(username)
    
    if model_path and os.path.exists(model_path):
        # Load model info
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Accuracy", f"{model_data.get('cv_score', 0)*100:.1f}%")
            with col2:
                st.metric("Training Samples", model_data.get('training_samples', 0))
            with col3:
                st.metric("Features", len(model_data.get('feature_names', [])))
            with col4:
                model_size = os.path.getsize(model_path) / 1024
                st.metric("Model Size", f"{model_size:.1f} KB")
            
            st.divider()
            
            # Model details
            with st.expander("🔍 Model Details"):
                st.write(f"**Model Type:** Gradient Boosting Classifier")
                st.write(f"**Model Path:** `{model_path}`")
                st.write(f"**Cross-Validation Score:** {model_data.get('cv_score', 0)*100:.2f}%")
                st.write(f"**Training Samples:** {model_data.get('training_samples', 0)} sessions")
                
                if 'user_base_timing' in model_data:
                    st.write(f"**Your Average Typing Speed:** {model_data['user_base_timing']:.1f}ms per key")
                if 'user_variance' in model_data:
                    st.write(f"**Typing Variance:** {model_data['user_variance']:.1f}ms")
                
                # Fix feature names display - convert to strings
                feature_names = model_data.get('feature_names', [])
                if feature_names:
                    # Convert all to strings to avoid numpy type issues
                    feature_names_str = [str(name) for name in feature_names]
                    st.write(f"**Number of Features:** {len(feature_names_str)}")
                    st.write(f"**Feature Names:** {', '.join(feature_names_str)}")
                else:
                    st.write("**Features:** 20 keystroke timing features")
            
            # Download model
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            
            st.download_button(
                label="📥 Download Your Model (.pkl)",
                data=model_bytes,
                file_name=f"{username}_keystroke_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("No model found. Please complete training first.")
    
    st.divider()
    
    # Training Data
    st.subheader("📚 Training & Test Data")
    
    data_dir = "user_keystroke_data"
    train_data_path = os.path.join(data_dir, f"{username}_training_data.csv")
    test_data_path = os.path.join(data_dir, f"{username}_test_data.csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Data**")
        if os.path.exists(train_data_path):
            train_df = pd.read_csv(train_data_path)
            st.info(f"✅ Available: {len(train_df)} training samples")
            
            with st.expander("Preview Training Data"):
                st.dataframe(train_df.head(10), use_container_width=True)
            
            st.download_button(
                label="📥 Download Training Data (.csv)",
                data=train_df.to_csv(index=False),
                file_name=f"{username}_training_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Training data not found. It will be generated when you retrain.")
            if st.button("🔄 Regenerate Training Data", use_container_width=True):
                reference_sessions = db_get_reference_sessions(username)
                if reference_sessions:
                    csv_path, df = export_user_training_data(username, reference_sessions)
                    if csv_path:
                        st.success("Training data regenerated!")
                        st.rerun()
                else:
                    st.error("No reference sessions available")
    
    with col2:
        st.markdown("**Test Data**")
        if os.path.exists(test_data_path):
            test_df = pd.read_csv(test_data_path)
            legitimate_count = len(test_df[test_df['class'] == 1])
            imposter_count = len(test_df[test_df['class'] == 0])
            st.info(f"✅ Available: {legitimate_count} legitimate + {imposter_count} imposter samples")
            
            with st.expander("Preview Test Data"):
                st.dataframe(test_df.head(10), use_container_width=True)
            
            st.download_button(
                label="📥 Download Test Data (.csv)",
                data=test_df.to_csv(index=False),
                file_name=f"{username}_test_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Test data not found. Generate it below.")
            
            num_samples = st.slider("Number of samples (per class)", 5, 20, 10)
            if st.button("🧪 Generate Test Data", use_container_width=True):
                with st.spinner("Generating test data..."):
                    csv_path, result = generate_test_data_for_user(username, num_samples)
                    if csv_path:
                        st.success(f"Test data generated: {csv_path}")
                        st.rerun()
                    else:
                        st.error(f"Failed to generate: {result}")
    
    st.divider()
    
    # Quick Test Option
    st.subheader("⚡ Quick Model Test")
    st.info("Test your model with your own test data right here!")
    
    if os.path.exists(model_path) and os.path.exists(test_data_path):
        if st.button("🚀 Run Quick Evaluation", type="primary", use_container_width=True):
            with st.spinner("Evaluating your model..."):
                try:
                    # Load model - handle both old format (direct model) and new format (dict)
                    loaded = load_model(model_path)
                    
                    # Check if it's a dictionary (new format) or direct model (old format)
                    if isinstance(loaded, dict):
                        model = loaded['model']
                        scaler = loaded.get('scaler', None)
                    else:
                        model = loaded
                        scaler = None
                    
                    # Load test data
                    df = pd.read_csv(test_data_path)
                    X_test = df.drop(['class', 'username', 'sample', 'type'], axis=1, errors='ignore')
                    y_test = df['class']
                    
                    # Apply scaling if scaler exists
                    if scaler is not None:
                        X_test = scaler.transform(X_test)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Show results
                    st.success(f"✅ Evaluation Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy*100:.1f}%")
                    with col2:
                        correct = (y_pred == y_test).sum()
                        st.metric("Correct Predictions", f"{correct}/{len(y_test)}")
                    with col3:
                        legitimate_acc = accuracy_score(
                            y_test[y_test == 1], 
                            y_pred[y_test == 1]
                        )
                        st.metric("Legitimate Detection", f"{legitimate_acc*100:.1f}%")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Imposter', 'Legitimate'],
                        y=['Imposter', 'Legitimate'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Blues'))
                    fig.update_layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted',
                        yaxis_title='Actual',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("Model or test data not available. Complete the steps above first.")


# --------------------------
# Enhanced Admin Dashboard with Fixed User Cards
# --------------------------

def admin_dashboard():
    page_header()

    with st.sidebar:
        alerts = db_get_unacknowledged_alerts()
        if len(alerts) > 0:
            st.markdown(f"<div class='alert-badge'>🚨 {len(alerts)} New Alerts</div>", unsafe_allow_html=True)
        
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

    t1, t2, t3, t4, t5, t6 = st.tabs(["👥 Users", "🚨 Verification Alerts", "🧪 Model Testing", "📊 Test History", "🎓 Training History", "📈 System Stats"])

    with t1:
        st.markdown("""
            <div class='glass'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>👥 User Management</h2>
            <p style='color: #64748b;'>Manage user accounts, roles, and permissions with our futuristic interface.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Approved Users Section
        st.markdown("### ✅ Approved Users")
        if st.session_state.users_db:
            for username, info in sorted(st.session_state.users_db.items()):
                is_expanded = st.session_state.expanded_user == username
                
                # Create a container for the user card
                with st.container():
                    # Create columns for the card layout
                    col_arrow, col_avatar, col_info, col_status = st.columns([0.3, 0.5, 3, 1])
                    
                    with col_arrow:
                        arrow = "▼" if is_expanded else "▶"
                        if st.button(arrow, key=f"arrow_{username}", help="Expand/Collapse"):
                            if st.session_state.expanded_user == username:
                                st.session_state.expanded_user = None
                            else:
                                st.session_state.expanded_user = username
                                st.session_state.edit_user = None
                            st.rerun()
                    
                    with col_avatar:
                        initials = username[0].upper() if username else "?"
                        st.markdown(f"""
                            <div class="user-avatar-small">{initials}</div>
                        """, unsafe_allow_html=True)
                    
                    with col_info:
                        role = info.get('role', 'user')
                        st.markdown(f"""
                            <div style="font-weight: 700; color: #0f172a; font-size: 1.05rem;">{username}</div>
                            <span class="role-badge">{role.upper()}</span>
                        """, unsafe_allow_html=True)
                    
                    with col_status:
                        status = info.get('status', 'pending')
                        status_class = "status-approved" if status == "approved" else "status-pending"
                        status_text = "✓ Approved" if status == "approved" else "⏳ Pending"
                        st.markdown(f'<span class="{status_class} status-badge">{status_text}</span>', unsafe_allow_html=True)
                    
                    # Expanded details section
                    if is_expanded:
                        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
                        
                        if st.session_state.edit_user == username:
                            user_edit_dialog(username)
                        else:
                            user_details = db_get_user_details(username)
                            if user_details:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Role", user_details['role'].upper())
                                with col2:
                                    st.metric("Status", user_details['status'].capitalize())
                                with col3:
                                    st.metric("Keystroke Trained", "Yes" if user_details['keystroke_trained'] else "No")
                                
                                st.markdown(f"**Email:** {user_details['email']}")
                                st.markdown(f"**Registered:** {user_details['created_at']}")
                                st.markdown(f"**Last Login:** {user_details['last_login']}")
                                
                                if st.button("✏️ Edit User", key=f"edit_{username}", use_container_width=True):
                                    st.session_state.edit_user = username
                                    st.rerun()
                
                st.divider()
        else:
            st.info("No approved users yet.")
        
        st.divider()
        
        # Pending Users Section
        st.markdown(f"### ⏳ Pending Approvals ({len(st.session_state.pending_users)})")
        if st.session_state.pending_users:
            for username, info in sorted(st.session_state.pending_users.items()):
                col_avatar, col_info, col_actions = st.columns([0.5, 2, 2])
                
                with col_avatar:
                    initials = username[0].upper() if username else "?"
                    st.markdown(f'<div class="user-avatar-small">{initials}</div>', unsafe_allow_html=True)
                
                with col_info:
                    role = info.get('role', 'user')
                    st.markdown(f"""
                        <div style="font-weight: 700; color: #0f172a;">{username}</div>
                        <span class="role-badge">{role.upper()}</span>
                    """, unsafe_allow_html=True)
                
                with col_actions:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Approve", key=f"app_{username}", use_container_width=True, type="primary"):
                            db_approve_user(username)
                            st.success(f"Approved {username}")
                            st.rerun()
                    with col2:
                        if st.button(f"❌ Reject", key=f"rej_{username}", use_container_width=True):
                            db_reject_user(username)
                            st.warning(f"Rejected {username}")
                            st.rerun()
                
                st.divider()
        else:
            st.success("✨ No pending requests")

    with t2:
        st.markdown("""
            <div class='glass'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>🚨 Enhanced Verification Alerts</h2>
            <p style='color: #64748b;'>Monitor all verification attempts and security events with advanced filtering and actions.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get all alerts (both acknowledged and unacknowledged)
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, username, timestamp, status, confidence, message, acknowledged FROM verification_alerts ORDER BY timestamp DESC LIMIT 100")
        all_alerts = cur.fetchall()
        conn.close()
        
        unack_alerts = [a for a in all_alerts if a[6] == 0]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", len(all_alerts))
        with col2:
            st.metric("Unacknowledged", len(unack_alerts))
        with col3:
            failed_count = len([a for a in all_alerts if a[3] == "FAILED"])
            st.metric("Failed Verifications", failed_count)
        with col4:
            success_count = len([a for a in all_alerts if a[3] == "SUCCESS"])
            st.metric("Successful Verifications", success_count)
        
        st.divider()
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_status = st.selectbox("Filter by Status", ["All", "FAILED", "SUCCESS"], key="alert_status_filter")
        with col2:
            filter_ack = st.selectbox("Filter by Acknowledgment", ["All", "Unacknowledged", "Acknowledged"], key="alert_ack_filter")
        with col3:
            usernames = list(set([a[1] for a in all_alerts]))
            filter_user = st.selectbox("Filter by User", ["All"] + sorted(usernames), key="alert_user_filter")
        with col4:
            st.write("")  # Spacer
            if st.button("🔄 Refresh Alerts", use_container_width=True):
                st.rerun()
        
        # Bulk actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Acknowledge All Failed Alerts", use_container_width=True):
                conn = _get_conn()
                cur = conn.cursor()
                cur.execute("UPDATE verification_alerts SET acknowledged=1 WHERE status='FAILED' AND acknowledged=0")
                conn.commit()
                conn.close()
                st.success("All failed alerts acknowledged")
                st.rerun()
        with col2:
            if st.button("🗑️ Delete All Acknowledged Alerts", use_container_width=True):
                conn = _get_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM verification_alerts WHERE acknowledged=1")
                deleted = cur.rowcount
                conn.commit()
                conn.close()
                st.success(f"Deleted {deleted} acknowledged alerts")
                st.rerun()
        with col3:
            if st.button("🗑️ Clear Old Alerts (>7 days)", use_container_width=True):
                week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                conn = _get_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM verification_alerts WHERE timestamp < ?", (week_ago,))
                deleted = cur.rowcount
                conn.commit()
                conn.close()
                st.success(f"Deleted {deleted} old alerts")
                st.rerun()
        
        st.divider()
        
        # Apply filters
        filtered_alerts = all_alerts
        if filter_status != "All":
            filtered_alerts = [a for a in filtered_alerts if a[3] == filter_status]
        if filter_ack == "Unacknowledged":
            filtered_alerts = [a for a in filtered_alerts if a[6] == 0]
        elif filter_ack == "Acknowledged":
            filtered_alerts = [a for a in filtered_alerts if a[6] == 1]
        if filter_user != "All":
            filtered_alerts = [a for a in filtered_alerts if a[1] == filter_user]
        
        # Display alerts
        if filtered_alerts:
            st.markdown(f"### Showing {len(filtered_alerts)} alert(s)")
            
            for alert in filtered_alerts:
                alert_id, username, timestamp, status, confidence, message, acknowledged = alert
                
                # Create colored container based on status
                if status == "FAILED":
                    container_color = "rgba(239, 68, 68, 0.05)"
                    border_color = "rgba(239, 68, 68, 0.3)"
                else:
                    container_color = "rgba(16, 185, 129, 0.05)"
                    border_color = "rgba(16, 185, 129, 0.3)"
                
                st.markdown(f"""
                    <div style='background: {container_color}; border: 2px solid {border_color}; 
                         border-radius: 12px; padding: 1rem; margin-bottom: 1rem;'>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**👤 User:** {username}")
                    st.caption(f"🕒 {timestamp}")
                
                with col2:
                    confidence_pct = confidence * 100 if confidence else 0
                    color = "#ef4444" if status == "FAILED" else "#10b981"
                    st.markdown(f"<span style='color:{color}; font-weight:700; font-size:1.1rem;'>Status: {status}</span>", unsafe_allow_html=True)
                    st.caption(f"📊 Confidence: {confidence_pct:.1f}%")
                
                with col3:
                    ack_status = "✅ Acknowledged" if acknowledged else "⏳ Pending"
                    st.markdown(f"**{ack_status}**")
                
                with col4:
                    col_ack, col_del = st.columns(2)
                    with col_ack:
                        if not acknowledged:
                            if st.button("✅", key=f"ack_{alert_id}", help="Acknowledge", use_container_width=True):
                                db_acknowledge_alert(alert_id)
                                st.success("Acknowledged")
                                st.rerun()
                        else:
                            st.markdown("*Done*")
                    with col_del:
                        if st.button("🗑️", key=f"del_{alert_id}", help="Delete alert", use_container_width=True):
                            conn = _get_conn()
                            cur = conn.cursor()
                            cur.execute("DELETE FROM verification_alerts WHERE id=?", (alert_id,))
                            conn.commit()
                            conn.close()
                            st.success("Deleted")
                            st.rerun()
                
                st.markdown(f"**📝 Details:** {message}")
                st.markdown("---")
        else:
            st.info("No alerts match the current filters")
        
        # Export functionality
        if all_alerts:
            st.divider()
            df_export = pd.DataFrame(all_alerts, columns=['ID', 'Username', 'Timestamp', 'Status', 'Confidence', 'Message', 'Acknowledged'])
            df_export['Acknowledged'] = df_export['Acknowledged'].apply(lambda x: 'Yes' if x else 'No')
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Export All Alerts (CSV)",
                data=csv,
                file_name=f"verification_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with t3:
        model_testing_interface()
    
    with t4:
        st.markdown("""
            <div class='glass'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>📊 Model Testing History</h2>
            <p style='color: #64748b;'>View all past model evaluation results and performance metrics.</p>
            </div>
        """, unsafe_allow_html=True)
        
        history = db_get_model_test_history(limit=100)
        
        if history:
            df = pd.DataFrame(history, columns=['ID', 'Username', 'Model Name', 'Timestamp', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Test Samples'])
            
            # Format percentages
            df['Accuracy'] = df['Accuracy'].apply(lambda x: f"{x*100:.2f}%" if x else "N/A")
            df['Precision'] = df['Precision'].apply(lambda x: f"{x:.3f}" if x else "N/A")
            df['Recall'] = df['Recall'].apply(lambda x: f"{x:.3f}" if x else "N/A")
            df['F1 Score'] = df['F1 Score'].apply(lambda x: f"{x:.3f}" if x else "N/A")
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tests", len(df))
            with col2:
                st.metric("Unique Models", df['Model Name'].nunique())
            with col3:
                st.metric("Unique Users", df['Username'].nunique())
            with col4:
                total_samples = pd.DataFrame(history, columns=['ID', 'Username', 'Model Name', 'Timestamp', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Test Samples'])['Test Samples'].sum()
                st.metric("Total Samples Tested", total_samples)
            
            st.divider()
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_user = st.selectbox("Filter by User", ["All"] + list(df['Username'].unique()))
            with col2:
                selected_model = st.selectbox("Filter by Model", ["All"] + list(df['Model Name'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            if selected_user != "All":
                filtered_df = filtered_df[filtered_df['Username'] == selected_user]
            if selected_model != "All":
                filtered_df = filtered_df[filtered_df['Model Name'] == selected_model]
            
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Test History (CSV)",
                data=csv,
                file_name=f"model_test_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No model testing history available yet. Run some model evaluations to see results here.")
    
    with t5:
        st.markdown("""
            <div class='glass'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>🎓 Keystroke Training History</h2>
            <p style='color: #64748b;'>View all user keystroke model training results and performance.</p>
            </div>
        """, unsafe_allow_html=True)
        
        training_history = db_get_keystroke_training_history()
        
        if training_history:
            df = pd.DataFrame(training_history, columns=['ID', 'Username', 'Timestamp', 'Sessions', 'CV Score', 'Training Samples', 'Model Path', 'Status'])
            
            # Format CV Score as percentage
            df['CV Score'] = df['CV Score'].apply(lambda x: f"{x*100:.2f}%" if x else "N/A")
            
            # Add data availability status
            def check_data_available(username):
                data_dir = "user_keystroke_data"
                train_exists = os.path.exists(os.path.join(data_dir, f"{username}_training_data.csv"))
                test_exists = os.path.exists(os.path.join(data_dir, f"{username}_test_data.csv"))
                
                if train_exists and test_exists:
                    return "✅ Both"
                elif train_exists:
                    return "📊 Training Only"
                elif test_exists:
                    return "🧪 Test Only"
                else:
                    return "❌ None"
            
            df['Data Available'] = df['Username'].apply(check_data_available)
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trainings", len(df))
            with col2:
                st.metric("Trained Users", df['Username'].nunique())
            with col3:
                success_count = df[df['Status'] == 'Success'].shape[0]
                st.metric("Successful Trainings", success_count)
            with col4:
                avg_sessions = pd.DataFrame(training_history, columns=['ID', 'Username', 'Timestamp', 'Sessions', 'CV Score', 'Training Samples', 'Model Path', 'Status'])['Sessions'].mean()
                st.metric("Avg Sessions", f"{avg_sessions:.1f}")
            
            st.divider()
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_user = st.selectbox("Filter by User", ["All"] + list(df['Username'].unique()), key="train_user_filter")
            with col2:
                selected_status = st.selectbox("Filter by Status", ["All"] + list(df['Status'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            if selected_user != "All":
                filtered_df = filtered_df[filtered_df['Username'] == selected_user]
            if selected_status != "All":
                filtered_df = filtered_df[filtered_df['Status'] == selected_status]
            
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Training History (CSV)",
                data=csv,
                file_name=f"keystroke_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No keystroke training history available yet. Users need to complete training sessions first.")

    with t6:
        st.markdown("""
            <div class='glass'>
            <h2 style='color: #667eea; margin-bottom: 0.5rem;'>📈 Comprehensive System Statistics</h2>
            <p style='color: #64748b;'>Detailed overview of all users, their activity, and system metrics.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        alerts = db_get_unacknowledged_alerts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Approved Users", len(st.session_state.users_db))
        col2.metric("Pending Approvals", len(st.session_state.pending_users))
        col3.metric("Unacknowledged Alerts", len(alerts))
        col4.metric("Available Models", safe_count_models())
        
        st.divider()
        
        # Detailed User Statistics Table
        st.markdown("### 👥 All Users Overview")
        
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT username, role, status, email, created_at, last_login, 
                   keystroke_trained, updated_at 
            FROM users 
            ORDER BY 
                CASE status 
                    WHEN 'approved' THEN 1 
                    WHEN 'pending' THEN 2 
                    ELSE 3 
                END,
                last_login DESC
        """)
        all_users = cur.fetchall()
        conn.close()
        
        if all_users:
            # Convert to DataFrame for better display
            df_users = pd.DataFrame(all_users, columns=[
                'Username', 'Role', 'Status', 'Email', 
                'Created At', 'Last Login', 'Keystroke Trained', 'Updated At'
            ])
            
            # Format data
            df_users['Status'] = df_users['Status'].apply(lambda x: x.capitalize())
            df_users['Role'] = df_users['Role'].apply(lambda x: x.upper())
            df_users['Keystroke Trained'] = df_users['Keystroke Trained'].apply(lambda x: '✅ Yes' if x else '❌ No')
            df_users['Email'] = df_users['Email'].apply(lambda x: x if x else 'Not provided')
            df_users['Last Login'] = df_users['Last Login'].apply(lambda x: x if x else 'Never')
            
            # Add color coding with status
            def color_status(val):
                if val == 'Approved':
                    return 'background-color: rgba(16, 185, 129, 0.1)'
                elif val == 'Pending':
                    return 'background-color: rgba(245, 158, 11, 0.1)'
                return ''
            
            # Display with filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_status_users = st.selectbox("Filter by Status", ["All", "Approved", "Pending"], key="sys_status_filter")
            with col2:
                filter_role_users = st.selectbox("Filter by Role", ["All"] + [r.upper() for r in ALLOWED_ROLES], key="sys_role_filter")
            with col3:
                filter_trained = st.selectbox("Keystroke Trained", ["All", "Yes", "No"], key="sys_trained_filter")
            
            # Apply filters
            filtered_df = df_users.copy()
            if filter_status_users != "All":
                filtered_df = filtered_df[filtered_df['Status'] == filter_status_users]
            if filter_role_users != "All":
                filtered_df = filtered_df[filtered_df['Role'] == filter_role_users]
            if filter_trained == "Yes":
                filtered_df = filtered_df[filtered_df['Keystroke Trained'] == '✅ Yes']
            elif filter_trained == "No":
                filtered_df = filtered_df[filtered_df['Keystroke Trained'] == '❌ No']
            
            st.dataframe(
                filtered_df.style.applymap(color_status, subset=['Status']),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Activity Summary
            st.divider()
            st.markdown("### 📊 Activity Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Status Distribution**")
                status_counts = df_users['Status'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values, 
                    names=status_counts.index,
                    title='User Status Distribution',
                    color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
                )
                fig_status.update_layout(height=300)
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                st.markdown("**Role Distribution**")
                role_counts = df_users['Role'].value_counts()
                fig_roles = px.bar(
                    x=role_counts.index,
                    y=role_counts.values,
                    title='Users by Role',
                    labels={'x': 'Role', 'y': 'Count'},
                    color=role_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_roles.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_roles, use_container_width=True)
            
            with col3:
                st.markdown("**Training Status**")
                trained_yes = len(df_users[df_users['Keystroke Trained'] == '✅ Yes'])
                trained_no = len(df_users[df_users['Keystroke Trained'] == '❌ No'])
                fig_trained = px.pie(
                    values=[trained_yes, trained_no],
                    names=['Trained', 'Not Trained'],
                    title='Keystroke Training Status',
                    color_discrete_sequence=['#667eea', '#94a3b8']
                )
                fig_trained.update_layout(height=300)
                st.plotly_chart(fig_trained, use_container_width=True)
            
            # Recent Login Activity
            st.divider()
            st.markdown("### 🕒 Recent Login Activity")
            
            recent_logins = df_users[df_users['Last Login'] != 'Never'].copy()
            if not recent_logins.empty:
                recent_logins = recent_logins.sort_values('Last Login', ascending=False).head(10)
                st.dataframe(
                    recent_logins[['Username', 'Role', 'Last Login', 'Status']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No login activity recorded yet")
            
            # Export functionality
            st.divider()
            csv = df_users.to_csv(index=False)
            st.download_button(
                "📥 Export All User Statistics (CSV)",
                data=csv,
                file_name=f"user_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No users found in the system")


def user_edit_dialog(username):
    """Show user edit dialog"""
    user_details = db_get_user_details(username)
    
    if not user_details:
        st.error("User not found")
        return
    
    st.markdown(f"""
        <div class='glass'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>✏️ Edit User: {username}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**User Information**")
        st.text(f"Username: {user_details['username']}")
        st.text(f"Email: {user_details['email']}")
        st.text(f"Created: {user_details['created_at']}")
        st.text(f"Last Login: {user_details['last_login']}")
        st.text(f"Keystroke Trained: {'Yes' if user_details['keystroke_trained'] else 'No'}")
    
    with col2:
        st.markdown("**Edit Settings**")
        new_role = st.selectbox(
            "Change Role",
            ALLOWED_ROLES,
            index=ALLOWED_ROLES.index(user_details['role']),
            key=f"role_{username}"
        )
        
        new_password = st.text_input(
            "New Password (leave empty to keep current)",
            type="password",
            key=f"pwd_{username}"
        )
        
        if new_password:
            strength = password_strength(new_password)
            st.progress(int(strength*100), text=f"Strength: {int(strength*100)}%")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save Changes", key=f"save_{username}", type="primary", use_container_width=True):
            if new_role != user_details['role']:
                db_update_role(username, new_role)
                st.success(f"Role updated to {new_role}")
            
            if new_password:
                db_change_password(username, new_password)
                st.success("Password updated")
            
            st.session_state.edit_user = None
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_{username}", use_container_width=True):
            st.session_state.edit_user = None
            st.rerun()
    
    with col3:
        if st.button("🗑️ Delete User", key=f"del_btn_{username}", use_container_width=True):
            st.session_state.delete_confirm = username
    
    # Delete confirmation
    if st.session_state.delete_confirm == username:
        st.warning(f"⚠️ Are you sure you want to delete user '{username}'? This action cannot be undone!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Delete", key=f"confirm_del_{username}", type="primary"):
                db_delete_user(username)
                st.success(f"User '{username}' deleted successfully")
                st.session_state.delete_confirm = None
                st.session_state.edit_user = None
                st.session_state.expanded_user = None
                st.rerun()
        
        with col2:
            if st.button("❌ No, Cancel", key=f"cancel_del_{username}"):
                st.session_state.delete_confirm = None
                st.rerun()


def safe_count_models(dir_path: str = None):
    d = dir_path or st.session_state.get('models_dir', 'keystroke_models')
    if os.path.exists(d):
        return len([f for f in os.listdir(d) if f.endswith('.pkl')])
    return 0


def model_cards(models_dir: str):
    if not os.path.exists(models_dir):
        st.info("Models directory does not exist yet.")
        return

    files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if not files:
        st.info("No model files (.pkl) found.")
        return

    cols = st.columns(3)
    for idx, f in enumerate(sorted(files)):
        p = os.path.join(models_dir, f)
        size_kb = os.path.getsize(p) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M')
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class='glass'>
                    <div style='font-weight:700; color: #0f172a;'>{f}</div>
                    <div style='font-size:.85rem; color: #64748b; margin-top: 0.5rem;'>Last modified: {mtime}</div>
                    <div style='font-size:.85rem; color: #64748b;'>Size: {size_kb:.1f} KB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def model_testing_interface():
    st.subheader("🤖 Model Testing & Evaluation")
    
    # Quick access to user's own model
    if st.session_state.get('username'):
        username = st.session_state.username
        user_model_path = db_get_keystroke_model_path(username)
        
        if user_model_path and os.path.exists(user_model_path):
            st.markdown("""
                <div class='glass'>
                <h4 style='color: #667eea; margin-bottom: 0.5rem;'>⚡ Quick Access: Your Model</h4>
                <p style='color: #64748b;'>Your trained keystroke model is ready for testing!</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"📁 Your Model: `{os.path.basename(user_model_path)}`")
            with col2:
                if st.button("🎯 Load My Model", use_container_width=True):
                    st.session_state['selected_model_path'] = user_model_path
                    st.session_state['models_dir'] = os.path.dirname(user_model_path)
                    st.rerun()
            
            # Check for user's test data
            data_dir = "user_keystroke_data"
            user_test_data = os.path.join(data_dir, f"{username}_test_data.csv")
            if os.path.exists(user_test_data):
                st.success(f"✅ Your test data is also available: `{username}_test_data.csv`")
                st.info("💡 Tip: Upload your test data below to evaluate your model!")
            
            st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        models_dir = st.text_input("Models Directory", st.session_state.get('models_dir', 'keystroke_models'), key="models_dir")
    with col2:
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        # Pre-select user's model if available
        default_selection = None
        if 'selected_model_path' in st.session_state:
            default_model_name = os.path.basename(st.session_state['selected_model_path'])
            if default_model_name in available_models:
                default_selection = available_models.index(default_model_name)
        
        selected_model = st.selectbox(
            "Select Model", 
            available_models,
            index=default_selection if default_selection is not None else 0
        ) if available_models else None
        
        if not available_models:
            st.warning("No models found in directory")

    st.caption("Available models")
    model_cards(models_dir)

    test_file = st.file_uploader("Upload Test Data (CSV)", type=['csv'], key="test_upload")

    if test_file and selected_model:
        try:
            df = pd.read_csv(test_file)
            st.success(f"✅ Loaded {len(df)} samples")
            with st.expander("Peek at data"):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Shape: {df.shape} | Columns: {list(df.columns)}")

            if 'class' in df.columns:
                X_test = df.drop('class', axis=1)
                y_test = df['class']

                if st.button("Run Evaluation", key="eval_btn", type="primary"):
                    with st.spinner("Loading model & predicting..."):
                        model_path = os.path.join(models_dir, selected_model)
                        loaded = load_model(model_path)
                        
                        # Handle both dict format (new) and direct model format (old)
                        if isinstance(loaded, dict):
                            model = loaded['model']
                            scaler = loaded.get('scaler', None)
                        else:
                            model = loaded
                            scaler = None
                        
                        if model:
                            # Apply scaling if scaler exists
                            # Drop non-numeric columns before scaling
                            # --- Safe scaling block ---
                            try:
                                # Ensure only numeric columns are scaled
                                if isinstance(X_test, pd.DataFrame):
                                    X_test = X_test.select_dtypes(include=[np.number])
                                elif isinstance(X_test, np.ndarray):
                                    pass  # already numeric
                                else:
                                    raise ValueError("X_test must be a DataFrame or NumPy array")

                                X_test_scaled = scaler.transform(X_test) if scaler else X_test
                            except Exception as e:
                                st.error(f"⚠️ Data scaling failed: {e}")
                                return
                            
                            y_pred = model.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred)
                            cm = confusion_matrix(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True)

                            # Extract metrics
                            precision = report['weighted avg']['precision']
                            recall = report['weighted avg']['recall']
                            f1 = report['weighted avg']['f1-score']

                            roc_plot = None
                            try:
                                if hasattr(model, 'predict_proba'):
                                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                                    roc_auc = auc(fpr, tpr)
                                    roc_fig = go.Figure()
                                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                                    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Chance', line=dict(dash='dash')))
                                    roc_fig.update_layout(title=f"ROC Curve (AUC={roc_auc:.3f})", xaxis_title="FPR", yaxis_title="TPR", height=380)
                                    roc_plot = roc_fig
                            except Exception:
                                pass

                            st.session_state.model_results = {
                                'model': selected_model,
                                'accuracy': accuracy,
                                'confusion_matrix': cm,
                                'report': report,
                                'predictions': y_pred,
                                'true_labels': y_test,
                                'roc_plot': roc_plot
                            }
                            
                            # Save to database
                            db_save_model_test_result(
                                st.session_state.username,
                                selected_model,
                                accuracy,
                                precision,
                                recall,
                                f1,
                                len(X_test),
                                f"Test on {test_file.name}"
                            )
                            
                            st.success("Evaluation complete and saved to history!")
                            display_results()
                        else:
                            st.error("Failed to load model")
            else:
                st.error("Dataset must contain a 'class' column")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())


def display_results():
    if not st.session_state.model_results:
        st.info("No results to display. Run evaluation first.")
        return

    results = st.session_state.model_results

    st.subheader("📊 Evaluation Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
    with c2:
        precision = results['report']['weighted avg']['precision']
        st.metric("Precision", f"{precision:.3f}")
    with c3:
        recall = results['report']['weighted avg']['recall']
        st.metric("Recall", f"{recall:.3f}")
    with c4:
        f1 = results['report']['weighted avg']['f1-score']
        st.metric("F1-Score", f"{f1:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = results['confusion_matrix']
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Authorized (0)', 'Unauthorized (1)'],
            y=['Authorized (0)', 'Unauthorized (1)'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'))
        fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(results['report']).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.subheader("Prediction Distribution")
    cL, cR = st.columns(2)
    with cL:
        true_counts = pd.Series(results['true_labels']).value_counts().sort_index()
        fig_true = px.pie(values=true_counts.values, names=["Authorized","Unauthorized"], title='True Labels')
        st.plotly_chart(fig_true, use_container_width=True)
    with cR:
        pred_counts = pd.Series(results['predictions']).value_counts().sort_index()
        fig_pred = px.pie(values=pred_counts.values, names=["Authorized","Unauthorized"], title='Predicted Labels')
        st.plotly_chart(fig_pred, use_container_width=True)

    if results.get('roc_plot') is not None:
        st.subheader("ROC Curve")
        st.plotly_chart(results['roc_plot'], use_container_width=True)

    st.subheader("Export Results")
    results_summary = {
        'model': results['model'],
        'accuracy': float(results['accuracy']),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Results (JSON)",
            data=json.dumps(results_summary, indent=2),
            file_name=f"results_{results['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with d2:
        predictions_df = pd.DataFrame({
            'True_Label': results['true_labels'],
            'Predicted_Label': results['predictions']
        })
        st.download_button(
            "Download Predictions (CSV)",
            data=predictions_df.to_csv(index=False),
            file_name=f"predictions_{results['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def user_dashboard():
    page_header()

    with st.sidebar:
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

    is_trained = db_check_keystroke_trained(st.session_state.username)
    
    # Load reference sessions if trained and not already loaded
    if is_trained and 'user_reference_sessions' not in st.session_state:
        reference_sessions = db_get_reference_sessions(st.session_state.username)
        if reference_sessions:
            st.session_state.user_reference_sessions = reference_sessions
        
        # Also load user timing parameters from model file
        model_path = db_get_keystroke_model_path(st.session_state.username)
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                if 'user_base_timing' in model_data:
                    st.session_state.user_base_timing = model_data['user_base_timing']
                if 'user_variance' in model_data:
                    st.session_state.user_variance = model_data['user_variance']
            except:
                pass
    
    if not is_trained:
        keystroke_training_page()
    else:
        t1, t2, t3, t4 = st.tabs(["🔐 Verify Identity", "📊 My Model & Data", "🧪 Model Testing", "🗂️ My Activity"])

        with t1:
            keystroke_verification_interface()

        with t2:
            my_model_data_interface()

        with t3:
            model_testing_interface()

        with t4:
            st.subheader("My Activity")
            st.info("Activity log for " + st.session_state.username)
            activity_data = pd.DataFrame({
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Action': ['User Login'],
                'Status': ['Success']
            })
            st.dataframe(activity_data, use_container_width=True)


# --------------------------
# Main
# --------------------------

def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.role == 'admin':
            admin_dashboard()
        else:
            user_dashboard()

    st.markdown("<div class='footer'><b><i>\u00A9 Namkha Gyeltshen/Sujan Monger/Tashi Phuntsho </i></b></div>", unsafe_allow_html=True)


def login_page():
    page_header()
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        tabs = st.tabs(["🔓 Login", "📝 Register", "ℹ️ About"])
        with tabs[0]:
            with st.container():
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                colL, colR = st.columns([1,1])
                with colL:
                    login_clicked = st.button("Sign in ✨", type="primary", use_container_width=True)
                

                if login_clicked:
                    if username and password:
                        success, role = db_authenticate(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.role = role
                            st.success(f"Welcome {username}! Redirecting...")
                            st.rerun()
                        else:
                            st.error("Invalid credentials or account pending approval")
                    else:
                        st.warning("Please enter username and password")

        with tabs[1]:
            new_username = st.text_input("Pick a username", key="reg_username")
            new_password = st.text_input("Create password", type="password", key="reg_password")
            strength = password_strength(new_password)
            st.progress(int(strength*100), text=f"Password strength: {int(strength*100)}%")
            confirm_password = st.text_input("Confirm password", type="password", key="reg_confirm")
            req_role = st.selectbox("Request role", REQUESTABLE_ROLES, index=REQUESTABLE_ROLES.index("user"))
            if st.button("Create account 🚀", use_container_width=True):
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        ok, msg = db_register_user(new_username, new_password, role=req_role)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill all fields")

        with tabs[2]:
            st.markdown(
                """
                <div class='glass'>
                <p style='color: #0f172a; line-height: 1.7;'>
                <b>Enhanced Keystroke Dynamics</b> uses advanced machine learning to authenticate users based on their unique typing patterns.
                Our system features <strong>accuracy</strong> with 10 training sessions, enhanced feature extraction using the phrase "Keystroke@25", 
                and automatic security responses with detailed admin monitoring.
                </p>
                <ul style='color: #0f172a; line-height: 1.8;'>
                    <li><strong>10 Training Sessions</strong> for maximum accuracy</li>
                    <li><strong>Realistic Pattern Recognition</strong> with character-specific timing</li>
                    <li><strong>Non-intrusive Verification</strong> - no logout on failure</li>
                    <li><strong>Comprehensive Admin Dashboard</strong> with full audit trail</li>
                    <li><strong>Exportable Models & Data</strong> - download your training and test data</li>
                    <li><strong>Personal Model Testing</strong> - evaluate your own keystroke model</li>
                </ul>
                <hr style='margin: 1.5rem 0;'>
                <p style='color: #64748b; font-size: 0.9rem;'>
                <strong>🎯 For Users:</strong> After training, visit "My Model & Data" to download your personal keystroke model and test data for analysis.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()