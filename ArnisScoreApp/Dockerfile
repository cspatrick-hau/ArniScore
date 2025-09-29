FROM python:3.11-slim

# Install system + GUI dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    xfce4 \
    x11vnc \
    xvfb \
    websockify \
    novnc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your app into the image
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose VNC + noVNC ports
EXPOSE 5900 6080

# Start virtual display + VNC + noVNC + your app
CMD Xvfb :0 -screen 0 1024x768x16 & \
    x11vnc -display :0 -nopw -listen 0.0.0.0 -xkb -forever & \
    websockify --web=/usr/share/novnc/ 6080 localhost:5900 & \
    export DISPLAY=:0 && python gui.py