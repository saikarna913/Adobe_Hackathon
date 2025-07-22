from pdf_feature_extractor import PDFFeatureExtractor

# Initialize the extractor
extractor = PDFFeatureExtractor("/workspaces/Adobe_Hackathon/Challenge_1a/Datasets/Pdfs/E0CCG5S239.pdf")

# Extract features
features_df = extractor.extract_features()

# Save to CSV
features_df.to_csv("raw_features.csv", index=False)