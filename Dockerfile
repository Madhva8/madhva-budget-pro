FROM python:3.9-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1-mesa-glx \
    libqt5gui5 \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    qtbase5-dev \
    qt5-qmake \
    libsecret-1-0 \
    libsecret-1-dev \
    libpam0g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set QT platform to offscreen for headless environments
ENV QT_QPA_PLATFORM=offscreen
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Create an entrypoint script for VNC support
RUN echo '#!/bin/bash\nxvfb-run -a python /app/src/main.py "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]