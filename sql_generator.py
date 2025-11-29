import csv
import os

# Set your file names
csv_filename = 'cards.csv'
output_filename = 'insert_statements.sql'

# Build absolute paths (adjust the folder as needed)
current_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
csv_file = os.path.join(current_dir, csv_filename)
output_file = os.path.join(current_dir, output_filename)

table_name = 'pkmn'
csv_to_db = {
    'TCGplayer Id': 'tcgplayer_id',
    'Product Line': 'product_line',
    'Set Name': 'set_name',
    'Product Name': 'product_name',
    'Title': 'title',
    'Number': 'number',
    'Rarity': 'rarity',
    'Condition': 'condition',
    'TCG Market Price': 'tcg_market_price',
    'TCG Direct Low': 'tcg_direct_low',
    'TCG Low Price With Shipping': 'tcg_low_shipped',
    'TCG Low Price': 'tcg_low',
    'Total Quantity': 'total_quantity',
    'Add to Quantity': 'add_quantity',
    'TCG Marketplace Price': 'tcg_marketplace_price',
    'Photo URL': 'photo_url'
}

# Columns in your table (matching CSV header)
columns = [
    'tcgplayer_id', 'product_line', 'set_name', 'product_name', 'title',
    'number', 'rarity', 'condition', 'tcg_market_price', 'tcg_direct_low',
    'tcg_low_shipped', 'tcg_low', 'total_quantity', 'add_quantity',
    'tcg_marketplace_price', 'photo_url'
]

def format_value(value):
    """Format a CSV value for SQLite INSERT"""
    value = value.strip()
    if value == '':
        return 'NULL'
    try:
        # Try to treat as number
        float_val = float(value)
        return value
    except ValueError:
        # Otherwise, treat as string
        # Escape single quotes in strings
        value = value.replace("'", "''")
        return f"'{value}'"

with open(csv_file, newline='', encoding='utf-8') as csvfile, open(output_file, 'w', encoding='utf-8') as sqlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        values = [format_value(row[csv_name]) for csv_name in csv_to_db.keys()]
        sql = f"INSERT INTO {table_name} ({', '.join(csv_to_db.values())}) VALUES ({', '.join(values)});"
        sqlfile.write(sql + '\n')

print(f"INSERT statements written to {output_file}")