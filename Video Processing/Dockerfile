FROM python:3.11

# Add a user
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Install system packages
COPY packages.txt /tmp/packages.txt
USER root
RUN apt-get update && \
    xargs -a /tmp/packages.txt apt-get install -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
USER user

# Set working directory
WORKDIR /app

# Install Python packages
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the code
COPY --chown=user . /app

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
