# PDF Outline Extraction Pipeline - Docker Usage Guide

This project provides a robust pipeline for extracting document outlines from PDFs using both machine learning and rule-based approaches. The entire environment is containerized for easy setup and reproducibility.

## Setup & Usage (Docker)

### 1. Build the Docker Image

Open a terminal in the project root (where the `Dockerfile` is located) and run:

```sh
docker build -t pdf-extractor .
```

This will create a Docker image named `pdf-extractor` with all dependencies and code pre-installed.

### 2. Prepare Your PDFs

Place your PDF files in the `pdf/` directory inside the project root.


### 3. Run the Pipeline

To process all PDFs in the `pdf/` directory, run: (mapping the local folders to docker volumes)

```sh
docker run --rm -v $(pwd)/pdf:/app/pdf -v $(pwd)/outputs:/app/outputs pdf-extractor
```

- On Windows (PowerShell):
  ```powershell
  docker run --rm -v ${PWD}/pdf:/app/pdf -v ${PWD}/outputs:/app/outputs pdf-extractor
  ```

All PDFs in the `pdf/` directory will be processed automatically.

### 4. View the Results

After the run, the combined outputs will be saved as `<pdfname>_output.json` in the `outputs/` directory (e.g., `outputs/sample_output.json`).

- Each output JSON contains the merged outline extracted by both the ML and rule-based extractors for that PDF.
- You can open these files with any text editor or JSON viewer.

---
