import pandas as pd
from datetime import datetime
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_timestamp(timestamp_str):
    """Convert string timestamp to datetime format"""
    try:
        # Remove trailing zeros (milliseconds)
        timestamp_str = str(timestamp_str)
        if len(timestamp_str) > 14:
            timestamp_str = timestamp_str[:14]
        
        return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    except ValueError as e:
        logging.error(f"Error converting timestamp {timestamp_str}: {e}")
        return None

def process_file(input_path, output_path=None):
    """Process a single CSV file"""
    try:
        # Read CSV without headers
        df = pd.read_csv(input_path, header=None)
        
        if len(df.columns) < 4:
            logging.error(f"File {input_path} has incorrect number of columns")
            return False

        # Convert first column to datetime
        df[0] = df[0].astype(str).apply(convert_timestamp)
        
        # Only keep the first 5 columns if there are more
        if len(df.columns) > 5:
            df = df.iloc[:, :5]
        
        # Add volume column if needed
        if len(df.columns) == 4:
            df[4] = 0  # Add volume column with zeros
        
        # Rename columns
        df.columns = ['timestamp', 'open', 'high', 'low', 'close']
        
        # Add volume column
        df['volume'] = 0
        
        # Generate output path if not provided
        if output_path is None:
            dir_name = os.path.dirname(input_path)
            file_name = os.path.basename(input_path)
            output_path = os.path.join(dir_name, f"converted_{file_name}")
        
        # Save to new CSV file
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully converted {input_path} to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        return False

def process_directory(input_dir, output_dir=None):
    """Process all CSV files in a directory"""
    if not os.path.exists(input_dir):
        logging.error(f"Input directory {input_dir} does not exist")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_count = 0
    fail_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename) if output_dir else None
            
            if process_file(input_path, output_path):
                success_count += 1
            else:
                fail_count += 1

    logging.info(f"Processing complete. Success: {success_count}, Failed: {fail_count}")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV timestamp format')
    parser.add_argument('input', help='Input CSV file or directory')
    parser.add_argument('--output', help='Output directory (optional)')
    
    args = parser.parse_args()

    if os.path.isfile(args.input):
        process_file(args.input, args.output)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        logging.error(f"Input path {args.input} does not exist")

if __name__ == "__main__":
    main() 