"""
Database service for expense tracking in DhanKanya.

This module provides database operations for storing and retrieving
expense tracking data per user.
"""

import sqlite3
import os
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Path to store user database (same as auth_service)
DB_DIR = "config"
DB_PATH = os.path.join(DB_DIR, "users.db")

def _ensure_expense_tables_exist():
    """Ensure the expense tracking tables exist in the database."""
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create transactions table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        description TEXT NOT NULL,
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        is_avoidable INTEGER,
        transaction_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create savings_goals table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS savings_goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        total_goal REAL NOT NULL,
        monthly_target REAL NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create voice_conversations table for storing voice interactions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS voice_conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_id TEXT,
        transcription TEXT NOT NULL,
        response_text TEXT NOT NULL,
        language_code TEXT NOT NULL,
        audio_duration REAL,
        interaction_mode TEXT DEFAULT 'voice_hybrid',
        created_at TEXT NOT NULL,
        metadata TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def get_user_id_by_username(username: str) -> Optional[int]:
    """
    Get user ID by username.
    
    Args:
        username: The username to look up
        
    Returns:
        The user ID if found, None otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else None

def save_voice_conversation(
    transcription: str,
    response_text: str,
    language_code: str,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    audio_duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save a voice conversation to the database.
    
    Args:
        transcription: User's transcribed speech
        response_text: AI's response text
        language_code: Language code ('ta', 'en', etc.)
        user_id: Optional user ID (for authenticated users)
        session_id: Optional session identifier
        audio_duration: Duration of audio in seconds
        metadata: Optional metadata dictionary
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Ensure tables exist
        _ensure_expense_tables_exist()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert metadata to JSON string if provided
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
        INSERT INTO voice_conversations 
        (user_id, session_id, transcription, response_text, language_code, 
         audio_duration, interaction_mode, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            session_id,
            transcription,
            response_text,
            language_code,
            audio_duration,
            'voice_hybrid',
            datetime.now().isoformat(),
            metadata_json
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Voice conversation saved for language: {language_code}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving voice conversation: {e}")
        return False

def get_voice_conversations(user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve voice conversations from the database.
    
    Args:
        user_id: Optional user ID to filter by
        limit: Maximum number of conversations to retrieve
        
    Returns:
        List of voice conversation dictionaries
    """
    try:
        # Ensure tables exist
        _ensure_expense_tables_exist()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
            SELECT * FROM voice_conversations 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
            SELECT * FROM voice_conversations 
            ORDER BY created_at DESC 
            LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert rows to dictionaries
        conversations = []
        for row in rows:
            conversation = {
                'id': row[0],
                'user_id': row[1],
                'session_id': row[2],
                'transcription': row[3],
                'response_text': row[4],
                'language_code': row[5],
                'audio_duration': row[6],
                'interaction_mode': row[7],
                'created_at': row[8],
                'metadata': json.loads(row[9]) if row[9] else {}
            }
            conversations.append(conversation)
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error retrieving voice conversations: {e}")
        return []

def save_transaction(
    user_id: int,
    transaction_date: date,
    description: str,
    amount: float,
    category: str,
    is_avoidable: Optional[bool],
    transaction_type: str
) -> bool:
    """
    Save a transaction to the database.
    
    Args:
        user_id: ID of the user
        transaction_date: Date of the transaction
        description: Description of the transaction
        amount: Amount of the transaction
        category: Category of the transaction
        is_avoidable: Whether the expense was avoidable (None for income)
        transaction_type: Type of transaction ('expense' or 'income')
        
    Returns:
        True if successful, False otherwise
    """
    _ensure_expense_tables_exist()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert date to string
        date_str = transaction_date.isoformat()
        created_at = datetime.now().isoformat()
        
        # Convert boolean to integer for SQLite
        is_avoidable_int = 1 if is_avoidable else 0 if is_avoidable is not None else None
        
        cursor.execute('''
        INSERT INTO transactions 
        (user_id, date, description, amount, category, is_avoidable, transaction_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, date_str, description, amount, category, is_avoidable_int, 
              transaction_type, created_at))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Transaction saved for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving transaction: {e}")
        return False

def get_transactions(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all transactions for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of transaction dictionaries
    """
    _ensure_expense_tables_exist()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM transactions
        WHERE user_id = ?
        ORDER BY date DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        transactions = []
        for row in rows:
            row_dict = dict(row)
            # Convert is_avoidable from int to bool
            if row_dict['is_avoidable'] is not None:
                row_dict['is_avoidable'] = bool(row_dict['is_avoidable'])
            transactions.append(row_dict)
        
        return transactions
    except Exception as e:
        logger.error(f"Error retrieving transactions: {e}")
        return []

def save_savings_goal(user_id: int, total_goal: float, monthly_target: float) -> bool:
    """
    Save a user's savings goal to the database.
    
    Args:
        user_id: ID of the user
        total_goal: Total savings goal amount
        monthly_target: Monthly savings target
        
    Returns:
        True if successful, False otherwise
    """
    _ensure_expense_tables_exist()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if goal already exists
        cursor.execute("SELECT id FROM savings_goals WHERE user_id = ?", (user_id,))
        existing_goal = cursor.fetchone()
        
        timestamp = datetime.now().isoformat()
        
        if existing_goal:
            # Update existing goal
            cursor.execute('''
            UPDATE savings_goals 
            SET total_goal = ?, monthly_target = ?, updated_at = ?
            WHERE user_id = ?
            ''', (total_goal, monthly_target, timestamp, user_id))
        else:
            # Insert new goal
            cursor.execute('''
            INSERT INTO savings_goals
            (user_id, total_goal, monthly_target, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (user_id, total_goal, monthly_target, timestamp, timestamp))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Savings goal saved for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving savings goal: {e}")
        return False

def get_savings_goal(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a user's savings goal.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dictionary with goal information if exists, None otherwise
    """
    _ensure_expense_tables_exist()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM savings_goals
        WHERE user_id = ?
        ''', (user_id,))
        
        goal = cursor.fetchone()
        conn.close()
        
        return dict(goal) if goal else None
    except Exception as e:
        logger.error(f"Error retrieving savings goal: {e}")
        return None 