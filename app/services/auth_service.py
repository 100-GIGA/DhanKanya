"""
Authentication service for DhanKanya.

This module provides functions for user authentication, including registration,
login, and user management using SQLite database.
"""

import hashlib
import secrets
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional, List, Union

# Configure logging
logger = logging.getLogger(__name__)

# Path to store user database
DB_DIR = "config"
DB_PATH = os.path.join(DB_DIR, "users.db")

def _ensure_db_exists():
    """Ensure the database and tables exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        created_at TEXT NOT NULL,
        last_login TEXT
    )
    ''')
    
    # Check if first_name and last_name columns exist, add them if they don't
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "first_name" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN first_name TEXT")
    
    if "last_name" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN last_name TEXT")
    
    conn.commit()
    conn.close()

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password with a salt using SHA-256.
    
    Args:
        password: The password to hash
        salt: Optional salt to use, generates a new one if None
        
    Returns:
        Tuple of (password_hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Combine password and salt, then hash
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

def register_user(username: str, email: str, password: str, first_name: str = "", last_name: str = "") -> bool:
    """
    Register a new user.
    
    Args:
        username: Username for the new account
        email: Email address for the new account
        password: Password for the new account
        first_name: First name of the user (optional)
        last_name: Last name of the user (optional)
        
    Returns:
        True if registration was successful, False otherwise
    """
    _ensure_db_exists()
    
    # Hash the password with a new salt
    password_hash, salt = _hash_password(password)
    created_at = datetime.now().isoformat()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert new user
        cursor.execute('''
        INSERT INTO users (username, email, password_hash, salt, first_name, last_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, salt, first_name, last_name, created_at))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User {username} registered successfully")
        return True
    except sqlite3.IntegrityError as e:
        logger.warning(f"Failed to register user: {e}")
        # Username or email already exists
        return False
    except Exception as e:
        logger.error(f"Error during user registration: {e}")
        return False

def authenticate_user(username_or_email: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user with username/email and password.
    
    Args:
        username_or_email: Username or email of the user
        password: Password to verify
        
    Returns:
        User data dictionary if authentication successful, None otherwise
    """
    _ensure_db_exists()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Find user by username or email
        cursor.execute('''
        SELECT * FROM users 
        WHERE username = ? OR email = ?
        ''', (username_or_email, username_or_email))
        
        user = cursor.fetchone()
        
        if not user:
            logger.warning(f"No user found with identifier: {username_or_email}")
            conn.close()
            return None
        
        # Get stored salt and hash the provided password
        salt = user['salt']
        password_hash, _ = _hash_password(password, salt)
        
        # Check if hashes match
        if password_hash != user['password_hash']:
            logger.warning(f"Invalid password for user: {username_or_email}")
            conn.close()
            return None
        
        # Update last login time
        last_login = datetime.now().isoformat()
        cursor.execute('''
        UPDATE users SET last_login = ? WHERE id = ?
        ''', (last_login, user['id']))
        
        conn.commit()
        
        # Convert user to dictionary
        user_dict = dict(user)
        user_dict['last_login'] = last_login
        
        conn.close()
        
        logger.info(f"User {user_dict['username']} authenticated successfully")
        return user_dict
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        return None

def get_user_profile(username: str) -> Optional[Dict]:
    """
    Get user profile data for display (excluding sensitive information).
    
    Args:
        username: Username of the user
        
    Returns:
        Dictionary with user profile data if user exists, None otherwise
    """
    _ensure_db_exists()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, username, email, first_name, last_name, created_at, last_login 
        FROM users 
        WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return None
        
        return dict(user)
    except Exception as e:
        logger.error(f"Error retrieving user profile: {e}")
        return None

def update_user_password(username: str, current_password: str, new_password: str) -> bool:
    """
    Update a user's password.
    
    Args:
        username: Username of the user
        current_password: Current password for verification
        new_password: New password to set
        
    Returns:
        True if password was updated successfully, False otherwise
    """
    _ensure_db_exists()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Find user
        cursor.execute('''
        SELECT id, password_hash, salt FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            logger.warning(f"No user found with username: {username}")
            conn.close()
            return False
        
        # Verify current password
        user_id, current_hash, salt = user
        password_hash, _ = _hash_password(current_password, salt)
        
        if password_hash != current_hash:
            logger.warning(f"Current password verification failed for user: {username}")
            conn.close()
            return False
        
        # Update password
        new_hash, new_salt = _hash_password(new_password)
        
        cursor.execute('''
        UPDATE users SET password_hash = ?, salt = ? WHERE id = ?
        ''', (new_hash, new_salt, user_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Password updated for user: {username}")
        return True
    except Exception as e:
        logger.error(f"Error updating password: {e}")
        return False

def update_user_profile(username: str, first_name: str = None, last_name: str = None, email: str = None) -> bool:
    """
    Update a user's profile information.
    
    Args:
        username: Username of the user to update
        first_name: New first name (optional)
        last_name: New last name (optional)
        email: New email (optional)
        
    Returns:
        True if update was successful, False otherwise
    """
    _ensure_db_exists()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Find user
        cursor.execute('''
        SELECT id FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            logger.warning(f"No user found with username: {username}")
            conn.close()
            return False
        
        user_id = user[0]
        updates = []
        params = []
        
        # Add update statements for provided fields
        if first_name is not None:
            updates.append("first_name = ?")
            params.append(first_name)
        
        if last_name is not None:
            updates.append("last_name = ?")
            params.append(last_name)
        
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        
        if not updates:
            logger.warning("No fields provided for update")
            conn.close()
            return False
        
        # Construct and execute update query
        update_query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        params.append(user_id)
        
        cursor.execute(update_query, params)
        conn.commit()
        conn.close()
        
        logger.info(f"Profile updated for user: {username}")
        return True
    except sqlite3.IntegrityError as e:
        # This could happen if email is already taken
        logger.warning(f"Failed to update profile: {e}")
        return False
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return False

def delete_user_account(username: str, password: str) -> bool:
    """
    Delete a user account after password verification.
    
    Args:
        username: Username of the account to delete
        password: Password for verification
        
    Returns:
        True if deletion was successful, False otherwise
    """
    _ensure_db_exists()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Find user
        cursor.execute('''
        SELECT id, password_hash, salt FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            logger.warning(f"No user found with username: {username}")
            conn.close()
            return False
        
        # Verify password
        user_id, current_hash, salt = user
        password_hash, _ = _hash_password(password, salt)
        
        if password_hash != current_hash:
            logger.warning(f"Password verification failed for user deletion: {username}")
            conn.close()
            return False
        
        # Delete user
        cursor.execute('''
        DELETE FROM users WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User account deleted: {username}")
        return True
    except Exception as e:
        logger.error(f"Error deleting user account: {e}")
        return False

# Migration function to transfer users from JSON to SQLite
def migrate_users_from_json():
    """Migrate users from the JSON file to SQLite if needed."""
    import json
    
    JSON_FILE = "config/users.json"
    
    if not os.path.exists(JSON_FILE):
        return
    
    try:
        # Read JSON data
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
            users = data.get("users", [])
        
        if not users:
            return
        
        # Ensure database exists
        _ensure_db_exists()
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert users
        for user in users:
            try:
                cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user["username"],
                    user["email"],
                    user["password_hash"],
                    user["salt"],
                    user["created_at"],
                    user.get("last_login")
                ))
            except sqlite3.IntegrityError:
                # Skip if user already exists
                pass
        
        conn.commit()
        conn.close()
        
        logger.info("Users migrated from JSON to SQLite successfully")
    except Exception as e:
        logger.error(f"Error migrating users from JSON: {e}")

# Run the migration on module import
migrate_users_from_json() 