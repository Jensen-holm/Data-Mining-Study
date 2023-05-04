# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 8080 for Render to proxy traffic to
EXPOSE 8080

# Start the Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "app:app"]
