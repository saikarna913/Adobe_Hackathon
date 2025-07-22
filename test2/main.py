import argparse
from pdf_feature_extractor import PDFFeatureExtractor
from model_trainer import OutlineModelTrainer
from outline_predictor import OutlinePredictor

def main():
    parser = argparse.ArgumentParser(description="PDF Outline Extraction System")
    subparsers = parser.add_subparsers(dest="command")
    
    # Extract features command
    extract_parser = subparsers.add_parser("extract", help="Extract features from PDF")
    extract_parser.add_argument("pdf_path", help="Path to input PDF")
    extract_parser.add_argument("--output", help="Path to save features CSV")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train outline detection model")
    train_parser.add_argument("data_path", help="Path to training data CSV")
    train_parser.add_argument("--save_model", help="Path to save trained model")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict outlines from PDF")
    predict_parser.add_argument("pdf_path", help="Path to input PDF")
    predict_parser.add_argument("model_path", help="Path to trained model")
    predict_parser.add_argument("--output", help="Path to save predictions CSV")
    predict_parser.add_argument("--threshold", type=float, default=0.5, 
                              help="Prediction probability threshold")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        extractor = PDFFeatureExtractor(args.pdf_path)
        features = extractor.extract_features()
        if args.output:
            features.to_csv(args.output, index=False)
            print(f"Features saved to {args.output}")
    
    elif args.command == "train":
        trainer = OutlineModelTrainer()
        trainer.train(args.data_path)
        if args.save_model:
            trainer.save_model(args.save_model)
    
    elif args.command == "predict":
        predictor = OutlinePredictor(args.model_path)
        outlines = predictor.predict_from_pdf(args.pdf_path, args.threshold)
        
        print("\nTop outline candidates:")
        for item in outlines[:10]:
            print(f"Page {item['page']+1} (Level {item.get('level', 0)}): {item['text']}")
        
        if args.output:
            pd.DataFrame(outlines).to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()