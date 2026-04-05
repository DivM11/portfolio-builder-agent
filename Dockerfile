# syntax=docker/dockerfile:1.4

# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /workspace

# Install Poetry
RUN pip install poetry

COPY pyproject.toml /workspace/portfolio-builder-agent/
COPY --from=agent_monitoring pyproject.toml /workspace/agent-monitoring/
COPY --from=agent_monitoring README.md /workspace/agent-monitoring/
COPY --from=agent_monitoring agent_monitoring /workspace/agent-monitoring/agent_monitoring

WORKDIR /workspace/portfolio-builder-agent

# Install dependencies using Poetry
# Using --no-root to avoid installing the project itself, as we are using Poetry only for dependency management
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . /workspace/portfolio-builder-agent

WORKDIR /workspace/portfolio-builder-agent

# Expose the main Streamlit application port
EXPOSE 8501


# Default command to run the application in app mode
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]