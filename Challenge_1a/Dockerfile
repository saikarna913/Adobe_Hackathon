FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the main script
CMD ["python", "process_pdfs.py"]
