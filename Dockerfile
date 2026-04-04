FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install dependencies first (for faster caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Hugging Face Spaces routes internal traffic to port 7860 by default
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
