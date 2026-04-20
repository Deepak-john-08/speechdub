#!/bin/bash
# SpeechDub - Quick Start Script

set -e

echo "==========================================="
echo "  SpeechDub — Multilingual AI Dubbing"
echo "==========================================="

# Check Python
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }

# Check ffmpeg
ffmpeg -version &>/dev/null || { echo "WARNING: ffmpeg not found. Install: sudo apt install ffmpeg"; }

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install minimal deps if django not found
python -c "import django" 2>/dev/null || {
    echo "Installing Django and core dependencies..."
    pip install --quiet django celery redis numpy
}

# Create media dirs
mkdir -p media/uploads media/processed media/dubbed media/jobs logs

# Run migrations
echo "Running database migrations..."
python manage.py migrate --run-syncdb 2>/dev/null || python manage.py migrate

# Check for superuser
echo ""
echo "Do you want to create a superuser for /admin/? (y/N)"
read -r create_super
if [[ "$create_super" == "y" || "$create_super" == "Y" ]]; then
    python manage.py createsuperuser
fi

echo ""
echo "==========================================="
echo "  Starting SpeechDub server..."
echo "  URL: http://127.0.0.1:8000"
echo ""
echo "  TIP: For background processing, open"
echo "  another terminal and run:"
echo "  source venv/bin/activate"
echo "  celery -A speechdub worker --loglevel=info"
echo "==========================================="
echo ""

python manage.py runserver
