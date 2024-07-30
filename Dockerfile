# Use a base image with TensorFlow and Python
FROM tensorflow/tensorflow:2.15.0

# Set environment variables
ENV FLASK_APP=application.py
ENV FLASK_RUN_HOST=0.0.0.0

# Create and set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip to avoid issues with distutils packages
RUN pip install --upgrade pip

# Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --ignore-installed --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000 to the outside world
# EXPOSE 5000

# Command to run the application
CMD ["flask", "run"]
