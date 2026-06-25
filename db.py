"""
SQLite database layer for DocVerify.

Schema matches Chapter 3 of the thesis:
  Table: documents  — document_id, file_path, upload_date, verification_status
  Table: results    — result_id, document_id, cnn_score, ocr_score, forensic_score, final_score
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("docverify.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path         TEXT    NOT NULL,
                original_filename TEXT    NOT NULL,
                upload_date       DATETIME DEFAULT CURRENT_TIMESTAMP,
                verification_status TEXT  NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results (
                result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id    INTEGER NOT NULL,
                cnn_score      REAL    NOT NULL,
                ocr_score      REAL    NOT NULL,
                forensic_score REAL    NOT NULL,
                final_score    REAL    NOT NULL,
                verdict        TEXT    NOT NULL,
                ocr_text       TEXT,
                ela_image      TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            );
        """)


def save_document(file_path: str, original_filename: str, status: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO documents (file_path, original_filename, verification_status) VALUES (?, ?, ?)",
            (file_path, original_filename, status),
        )
        return cur.lastrowid


def save_result(
    document_id: int,
    cnn_score: float,
    ocr_score: float,
    forensic_score: float,
    final_score: float,
    verdict: str,
    ocr_text: str = "",
    ela_image: str = "",
) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO results
               (document_id, cnn_score, ocr_score, forensic_score, final_score, verdict, ocr_text, ela_image)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (document_id, cnn_score, ocr_score, forensic_score, final_score, verdict, ocr_text, ela_image),
        )


def get_document_with_result(document_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            """SELECT d.*, r.cnn_score, r.ocr_score, r.forensic_score,
                      r.final_score, r.verdict, r.ocr_text, r.ela_image
               FROM documents d
               LEFT JOIN results r ON d.document_id = r.document_id
               WHERE d.document_id = ?""",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def get_all_documents(limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT d.document_id, d.original_filename, d.upload_date,
                      d.verification_status, r.final_score
               FROM documents d
               LEFT JOIN results r ON d.document_id = r.document_id
               ORDER BY d.upload_date DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
