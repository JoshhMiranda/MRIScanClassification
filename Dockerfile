# FROM tensorflow/tensorflow:2.15.0

# # Set environment variables
# ENV FLASK_APP=application.py
# ENV FLASK_RUN_HOST=0.0.0.0

# # Create and set the working directory
# WORKDIR /app

# # Copy the requirements file into the container
# COPY requirements.txt .

# # Upgrade pip to avoid issues with distutils packages
# RUN pip install --upgrade pip

# # Install dependencies
# # RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --ignore-installed --no-cache-dir -r requirements.txt

# # Copy the rest of the application code into the container
# COPY . .



# # Command to run the application
# CMD ["flask", "run"]


FROM python:3.10.13-slim

WORKDIR /app

# Copy all files into /app
COPY . /app

# Update package list and install necessary packages
RUN apt-get update -y && \
    apt-get install -y awscli ffmpeg libsm6 libxext6 unzip && \
    pip install -r requirements.txt

# # Verify the file exists and permissions
# RUN ls -l /app
# RUN python3 -c "import os; print(os.path.isfile('/app/application.py'))"

# Expose port 5000 to the outside world
EXPOSE 8080

# Run the application
CMD ["python3", "app.py", "--host", "0.0.0.0"]
