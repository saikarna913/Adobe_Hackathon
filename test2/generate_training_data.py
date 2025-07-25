import os
import pandas as pd
import json
from pdf_feature_extractor import PDFFeatureExtractor
import sys

def generate_training_data(pdf_dir, json_dir, output_path):
    """
    Generates a consolidated training data file (XLSX) from a directory of PDFs
    and their corresponding ground truth JSON files.

    Args:
        pdf_dir (str): Path to the directory containing PDF files.
        json_dir (str): Path to the directory containing JSON ground truth files.
        output_path (str): Path to save the output XLSX file.
    """
    all_features = []
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_path = os.path.join(json_dir, os.path.splitext(pdf_file)[0] + '.json')
        
        if not os.path.exists(json_path):
            print(f"⚠️ Warning: No JSON found for {pdf_file}, skipping.")
            continue
            
        print(f"Processing: {pdf_file}")
        
        # Extract features
        try:
            extractor = PDFFeatureExtractor(pdf_path)
            df = extractor.extract_features()
        except Exception as e:
            print(f"Error extracting features from {pdf_file}: {e}")
            continue

        # Load ground truth
        with open(json_path, 'r') as f:
            ground_truth = json.load(f)
        
        outline_texts = {item['text'].strip() for item in ground_truth.get('outline', [])}
        
        # Label the data
        df['is_outline'] = df['text'].apply(lambda x: 1 if x.strip() in outline_texts else 0)
        
        all_features.append(df)
        
    if not all_features:
        print("No features were extracted. Aborting.")
        return

    # Combine all dataframes
    final_df = pd.concat(all_features, ignore_index=True)
    
    # Save to Excel
    final_df.to_excel(output_path, index=False)
    print(f"✅ Training data successfully generated at: {output_path}")
    print(f"Total rows: {len(final_df)}, Total outlines: {final_df['is_outline'].sum()}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_training_data.py <pdf_directory> <json_directory> <output_xlsx_path>")
        sys.exit(1)
        
    pdf_dir = sys.argv[1]
    json_dir = sys.argv[2]
    output_path = sys.argv[3]
    
    generate_training_data(pdf_dir, json_dir, output_path)
