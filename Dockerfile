# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

COPY pyproject.toml /app/

# Install dependencies using Poetry
# Using --no-root to avoid installing the project itself, as we are using Poetry only for dependency management
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . /app

# Expose Streamlit (main app), monitoring API and monitoring UI ports
EXPOSE 8501
EXPOSE 8000
EXPOSE 8502


# Default command to run the application in app mode
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]