## This is my first attempt to automatically containarize my repo using github actions.

# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements/requirements_pip.txt

# Set the command to run your Python application
CMD ["python", "main_glen_bert.py"]
