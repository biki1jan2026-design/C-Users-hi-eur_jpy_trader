FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PySide6 and display
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY ../backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY . .

# Set environment for headless operation (if needed)
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:0

# Expose web interface port (if using web-based GUI)
EXPOSE 3000

# Default command - note: PySide6 GUI requires display
# For headless servers, use the backend-only container
CMD ["python", "-c", "print('GUI container requires display. Use backend container for headless operation.')"]
