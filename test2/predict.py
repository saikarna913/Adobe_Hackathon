import pandas as pd
import joblib
from model_trainer import OutlineModelTrainer  # adjust if filename is different

# Load your test data
test_df = pd.read_excel("test1_features.xlsx")

# Define the same features used during training
selected_features = [
        'relative_font_size', 'is_bold', 'is_title_case',
        'ends_with_punct', 'starts_with_number', 
         'relative_y', 'length_norm',
        'line_density', 'prev_spacing', 'next_spacing','outline_score',	
        'heading_score',	'title_score',	'norm_font_size'	,'norm_indent'

    ]

# Initialize trainer and load model
trainer = OutlineModelTrainer(feature_names=selected_features)
trainer.load_model("Training_data_rf_model.pkl")  # update path as needed

# Predict
_, predicted_labels = trainer.predict(test_df)

# Attach predictions to your dataframe
test_df["predicted_label"] = predicted_labels

# Save or inspect results
test_df.to_excel("predicted_output.xlsx", index=False)
print(test_df[["text", "predicted_label"]].head())
