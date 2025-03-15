# Use an official Python 3.10.12 base image
FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .
COPY resnet50_dtd_split1.pth .
COPY TBot_resnet50.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, for debugging)
EXPOSE 8080

# Set environment variables (if needed)
ENV TELEGRAM_TOKEN="your_telegram_token"

# Run the bot
CMD ["python3", "TBot_resnet50.py"]
