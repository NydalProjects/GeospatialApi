FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app

RUN apt-get update && apt-get install -y \
    libexpat1 \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . /app


# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container

# Run app.py when the container launches
CMD ["gunicorn", "-w", "8", "--threads", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "main:app"]
