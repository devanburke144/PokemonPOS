import pdb

import numpy as np
import csv
import os
import datetime

import sqlite3
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cards.csv")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
DB_PATH = os.path.join(BASE_DIR, "database.db")
SKU_PATH = os.path.join(BASE_DIR, "sku.txt")
TABLE_NAME = 'pkmn' # 'inventory'
LOG_FOLDER = "logs"

# Example table schema (customize this)
# created_at TEXT DEFAULT CURRENT_TIMESTAMP
TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pkmn(
    tcgplayer_id TEXT,
    product_line TEXT,
    set_name TEXT,
    product_name TEXT,
    title TEXT,
    number TEXT,
    rarity TEXT,
    condition TEXT,
    tcg_market_price REAL,
    tcg_direct_low REAL,
    tcg_low_shipped REAL,
    tcg_low REAL,
    total_quantity INTEGER,
    add_quantity INTEGER,
    tcg_marketplace_price REAL,
    photo_url TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

INVENTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory(
    tcgplayer_id TEXT,
    product_line TEXT,
    set_name TEXT,
    product_name TEXT,
    title TEXT,
    number TEXT,
    rarity TEXT,
    condition TEXT,
    tcg_market_price REAL,
    tcg_direct_low REAL,
    tcg_low_shipped REAL,
    tcg_low REAL,
    total_quantity INTEGER,
    add_quantity INTEGER,
    tcg_marketplace_price REAL,
    photo_url TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def get_connection():
    """Create and return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def create_table():
    """Create the main table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(TABLE_SCHEMA)
        conn.commit()

def insert_record(data: Dict[str, Any]) -> int:
    """
    Insert a record with dynamically provided fields.
    Returns the inserted row ID.

    Example:
        insert_record({"name": "Sword", "value": 10})
    """
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?" for _ in data])
    values = tuple(data.values())

    query = f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"

    with get_connection() as conn:
        cursor = conn.execute(query, values)
        conn.commit()
        return cursor.lastrowid

def fetch_all(where: Dict[str, Any] = None) -> List[Tuple]:
    """
    Fetch all rows, with optional filtering.

    Example:
        fetch_all({"name": "Sword"})
    """
    if where:
        clauses = " AND ".join([f"{k}=?" for k in where.keys()])
        query = f"SELECT * FROM {TABLE_NAME} WHERE {clauses}"
        params = tuple(where.values())
    else:
        query = f"SELECT * FROM {TABLE_NAME}"
        params = ()

    with get_connection() as conn:
        return conn.execute(query, params).fetchall()

def update_record(tcgplayer_id: str, updates: Dict[str, Any]) -> None:
    """
    Update fields in a record identified by tcgplayer_id.

    Example:
        update_record("8816217", {"tcg_market_price": 12.50, "rarity": "Ultra Rare"})
    """
    if not updates:
        return  # nothing to update

    set_clause = ", ".join([f"{k}=?" for k in updates.keys()])
    values = tuple(updates.values()) + (tcgplayer_id,)

    query = f"UPDATE {TABLE_NAME} SET {set_clause} WHERE tcgplayer_id=?"

    with get_connection() as conn:
        conn.execute(query, values)
        conn.commit()


def delete_record(tcgplayer_id: str) -> None:
    """
    Delete a record identified by tcgplayer_id.
    """
    with get_connection() as conn:
        conn.execute(
            f"DELETE FROM {TABLE_NAME} WHERE tcgplayer_id=?",
            (tcgplayer_id,)
        )
        conn.commit()

def record_exists(tcgplayer_id: str) -> bool:
    with get_connection() as conn:
        result = conn.execute(
            f"SELECT 1 FROM {TABLE_NAME} WHERE tcgplayer_id=? LIMIT 1",
            (tcgplayer_id,)
        ).fetchone()
        return result is not None

def view_schema() -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns the schema of the database.
    Dictionary format:
        {
            "table_name": [
                ("column_name", "column_type"),
                ...
            ]
        }

    Prints the schema to the console.
    """
    schema_info = {}

    with get_connection() as conn:
        cursor = conn.cursor()

        # Fetch all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()  # cid, name, type, notnull, dflt_value, pk

            schema_info[table] = [(col[1], col[2]) for col in columns]

        # Pretty print the schema
        print("\n===== DATABASE SCHEMA =====")
        for table, cols in schema_info.items():
            print(f"\nTable: {table}")
            for name, col_type in cols:
                print(f"  - {name}: {col_type}")
        print("\n============================\n")

    return schema_info

def record_to_string_pretty(record, columns=[
    "tcgplayer_id",
    "product_line",
    "set_name",
    "product_name",
    "title",
    "number",
    "rarity",
    "condition",
    "tcg_market_price",
    "tcg_direct_low",
    "tcg_low_shipped",
    "tcg_low",
    "total_quantity",
    "add_quantity",
    "tcg_marketplace_price",
    "photo_url",
    "created_at"
]):
    """
    Pretty-prints a tuple record. If no columns list is provided,
    column names default to col1, col2, col3, ...
    """
    if not record:
        return "<empty record>"

    # Auto-generate column names if none are provided
    if columns is None:
        columns = [f"col{i+1}" for i in range(len(record))]

    longest_col = max(len(col) for col in columns)

    lines = []
    for col, val in zip(columns, record):
        lines.append(f"{col.ljust(longest_col)} : {val}")

    return "\n".join(lines)

round_point4 = lambda x: np.floor(x + 0.6)
def custom_round_function(num):
    return f'${round_point4(num / 5) * 5 if num >= 100 else round_point4(num):.2f}'

def master_checker(sku):
    result = fetch_all({"tcgplayer_id" : f"{sku}"})
    if result:
        card_name = result[0][3] # can be NULL
        market_price = result[0][8] # can be NULL
        return [sku, card_name, market_price]
    return [sku, -1, -1] # card DNE
    
def update_csv(csv_path=CSV_PATH):
    data_cleaned = np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1)
  
    quantities = np.array(data_cleaned[:,13].astype(int))
    data_cleaned = data_cleaned[:, [0,3,8]]
    data_cleaned = np.repeat(data_cleaned, quantities, 0)
    vector_rounder = np.vectorize(custom_round_function)
    data_cleaned[:, 2] = np.array([master_checker(sku)[2] for sku in data_cleaned[:, 0]]) # pull prices from database instead
    data_cleaned[:, 2] = vector_rounder(data_cleaned[:, 2].astype('float64')).astype('<U65')
    d_left = data_cleaned[0::2]
    d_right = data_cleaned[1::2]
    d_right = np.pad(d_right, (0,d_left.shape[0]-d_right.shape[0]))
    data_cleaned = np.hstack((d_left, d_right))


    with open(DATA_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_cleaned)

def update_csv_from_sku_file(sku_path=SKU_PATH):
    data_cleaned = np.genfromtxt(sku_path, delimiter=",", dtype=str, skip_header=0)
    data_cleaned = np.array([master_checker(sku) for sku in data_cleaned])
    #quantities = np.array(data_cleaned[:,13].astype(int))
    # data_cleaned = data_cleaned[:, [0,3,8]]
    #data_cleaned = np.repeat(data_cleaned, quantities, 0)
    vector_rounder = np.vectorize(custom_round_function)
    data_cleaned[:, 2] = vector_rounder(data_cleaned[:, 2].astype('float64')).astype('<U65')
    d_left = data_cleaned[0::2]
    d_right = data_cleaned[1::2]
    d_right = np.pad(d_right, (0,d_left.shape[0]-d_right.shape[0]))
    data_cleaned = np.hstack((d_left, d_right))[:, :-1]


    with open(DATA_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_cleaned)

def scan_cards():
    """
    Loop to scan multiple tcgplayer_id values.
    Prints each matching card and accumulates total cost.
    Saves a CSV log to /logs folder including sale price.
    """
    all_card_records = []  # (id, name, price)

    print("=== Card Scanner ===")
    print("Enter tcgplayer_id values. Type 'done' to finish.\n")

    while True:
        sku = input("Scan ID: ").strip()

        if sku.lower() in ("done", "exit", "quit") or sku == "":
            break

        rows = fetch_all({"tcgplayer_id": sku})

        if not rows:
            print(f"[!] No card found for ID {sku}\n")
            continue

        row = rows[0]
        print(record_to_string_pretty(row))
        print()

        card_name = row[3]
        price = row[8]

        if price is None or price == "":
            print(f"[!] Warning: card {sku} has no market price.\n")
            continue

        try:
            price_float = float(price)
        except ValueError:
            print(f"[!] Price for {sku} is invalid: {price}\n")
            continue

        rounded_str = custom_round_function(price_float)  # e.g. "$12.00"
        rounded_price = float(rounded_str.replace("$", ""))

        all_card_records.append((sku, card_name, rounded_price))

    # Finished
    total_cost = sum(price for (_, _, price) in all_card_records)

    print("\n=== Scanning Complete ===")
    print(f"Total market cost of scanned cards: ${total_cost:.2f}")

    for sku, name, price in all_card_records:
        print(f"{name:<30} : ${price:.2f}")

    print("==========================\n")

    # ----------------------------------------------
    #        OPTIONAL SALE PRICE
    # ----------------------------------------------
    sale_price = None
    while True:
        user_input = input("Enter sale price for this transaction (or leave blank): ").strip()
        if user_input == "":
            break
        try:
            sale_price = float(user_input)
            break
        except ValueError:
            print("[!] Invalid number. Try again.")

    # ----------------------------------------------
    #        CREATE LOGS FOLDER IF MISSING
    # ----------------------------------------------
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)

    # ----------------------------------------------
    #        WRITE LOG FILE
    # ----------------------------------------------

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_log_{timestamp}.csv"
    filepath = os.path.join(LOG_FOLDER, filename)

    rows = [("tcgplayer_id", "card_name", "market_price")]
    for sku, name, price in all_card_records:
        rows.append((sku, name, f"{price:.2f}"))

    rows.append(("TOTAL_MARKET", "", f"{total_cost:.2f}"))
    rows.append(("SALE_PRICE", "", "" if sale_price is None else f"{sale_price:.2f}"))

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[+] CSV log saved to: {filepath}\n")


if __name__ == "__main__":
    update_csv()