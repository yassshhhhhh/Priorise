import sqlite3
import json
from datetime import datetime
import os

class DocumentDatabase:
    def __init__(self, db_path="documents.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create table for processed documents
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_documents (
                    filename TEXT PRIMARY KEY,
                    file_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    processed_date TEXT,
                    file_hash TEXT
                )
            ''')
            
            conn.commit()

    def save_document(self, filename, file_type, content, metadata=None, file_hash=None):
        """Save processed document data to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert metadata to JSON string if it's a dict
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            # Convert content to JSON string if it's a dict
            if isinstance(content, dict):
                content = json.dumps(content)
            
            cursor.execute('''
                INSERT OR REPLACE INTO processed_documents 
                (filename, file_type, content, metadata, processed_date, file_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                file_type,
                content,
                metadata,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_hash
            ))
            
            conn.commit()

    def get_document(self, filename):
        """Retrieve processed document data from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_type, content, metadata, processed_date, file_hash
                FROM processed_documents
                WHERE filename = ?
            ''', (filename,))
            
            result = cursor.fetchone()
            
            if result:
                file_type, content, metadata, processed_date, file_hash = result
                
                # Parse JSON strings back to dictionaries
                try:
                    content = json.loads(content)
                except:
                    pass
                
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
                
                return {
                    "file_type": file_type,
                    "content": content,
                    "metadata": metadata,
                    "processed_date": processed_date,
                    "file_hash": file_hash
                }
            
            return None

    def delete_document(self, filename):
        """Delete a document from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM processed_documents
                WHERE filename = ?
            ''', (filename,))
            
            conn.commit()

    def get_all_documents(self):
        """Retrieve all processed documents from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT filename, file_type, content, metadata, processed_date, file_hash
                FROM processed_documents
            ''')
            
            results = cursor.fetchall()
            
            documents = []
            for result in results:
                filename, file_type, content, metadata, processed_date, file_hash = result
                
                # Parse JSON strings back to dictionaries
                try:
                    content = json.loads(content)
                except:
                    pass
                
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
                
                documents.append({
                    "filename": filename,
                    "file_type": file_type,
                    "content": content,
                    "metadata": metadata,
                    "processed_date": processed_date,
                    "file_hash": file_hash
                })
            
            return documents 