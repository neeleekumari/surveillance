# Deployment Guide

This guide provides instructions for deploying the Floor Monitoring Application in a production environment.

## System Requirements

### Hardware Requirements

- **CPU**: Intel i5 or equivalent (Intel i7 recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free disk space
- **Graphics**: DirectX 11 compatible GPU (for GPU acceleration)
- **Cameras**: USB cameras (1 or more)
- **Network**: Ethernet connection recommended for database access

### Software Requirements

- **Operating System**: 
  - Windows 10/11 (64-bit)
  - Ubuntu 18.04+ (64-bit)
  - macOS 10.15+ (64-bit)
- **Python**: 3.10 or higher
- **Database**: PostgreSQL 14 or higher
- **Dependencies**: As listed in `requirements.txt`

## Installation

### 1. Prepare the Environment

#### Windows

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/)
3. Install Git from [git-scm.com](https://git-scm.com/downloads)

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Install Git
sudo apt install git

# Install system dependencies for OpenCV
sudo apt install libgl1 libglib2.0-0
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Install PostgreSQL
brew install postgresql

# Install Git
brew install git
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd floor_monitoring_app
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 5. Set Up PostgreSQL Database

#### Create Database User and Database

```bash
# Switch to postgres user (Linux/macOS)
sudo -u postgres psql

# Or connect directly (Windows)
psql -U postgres
```

```sql
-- Create database user
CREATE USER floor_monitor WITH PASSWORD 'your_secure_password';

-- Create database
CREATE DATABASE floor_monitor OWNER floor_monitor;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE floor_monitor TO floor_monitor;

-- Exit psql
\q
```

#### Initialize Database Tables

The application will automatically create tables on first run, or you can manually create them:

```sql
-- Connect to floor_monitor database
\c floor_monitor

-- Create workers table
CREATE TABLE workers (
    worker_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(100),
    contact VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create activity_log table
CREATE TABLE activity_log (
    log_id SERIAL PRIMARY KEY,
    worker_id INTEGER REFERENCES workers(worker_id),
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds INTEGER
);
```

### 6. Configure the Application

Edit `config/config.json`:

```json
{
    "database": {
        "host": "localhost",
        "name": "floor_monitor",
        "user": "floor_monitor",
        "password": "your_secure_password",
        "port": 5432
    },
    "cameras": [
        {
            "id": 0,
            "name": "Main Entrance"
        },
        {
            "id": 1,
            "name": "Production Floor"
        }
    ],
    "thresholds": {
        "warning_minutes": 15,
        "alert_minutes": 30
    },
    "notifications": {
        "enabled": true,
        "sound": true
    },
    "app": {
        "version": "1.0.0",
        "debug": false
    }
}
```

### 7. Test the Installation

```bash
# Run the application
python run.py
```

## Production Deployment

### 1. Create Production Configuration

Create a separate configuration for production:

```bash
cp config/config.json config/config.prod.json
```

Edit `config/config.prod.json` with production settings:

```json
{
    "database": {
        "host": "your-production-db-host",
        "name": "floor_monitor",
        "user": "floor_monitor",
        "password": "your_production_password",
        "port": 5432
    },
    "cameras": [
        {
            "id": 0,
            "name": "Main Entrance"
        }
    ],
    "thresholds": {
        "warning_minutes": 15,
        "alert_minutes": 30
    },
    "notifications": {
        "enabled": true,
        "sound": false
    },
    "app": {
        "version": "1.0.0",
        "debug": false
    }
}
```

### 2. Set Up as a Service (Linux)

Create a systemd service file `/etc/systemd/system/floor-monitor.service`:

```ini
[Unit]
Description=Floor Monitoring Application
After=network.target postgresql.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/floor_monitoring_app
Environment=PYTHONPATH=/path/to/floor_monitoring_app/src
ExecStart=/path/to/floor_monitoring_app/venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable floor-monitor.service
sudo systemctl start floor-monitor.service
```

### 3. Set Up as a Service (Windows)

Create a Windows service using NSSM (Non-Sucking Service Manager):

1. Download NSSM from [nssm.cc](https://nssm.cc/download)
2. Install as a service:

```cmd
nssm install FloorMonitor "C:\path\to\floor_monitoring_app\venv\Scripts\python.exe" "C:\path\to\floor_monitoring_app\run.py"
nssm start FloorMonitor
```

### 4. Deploy Executable Package

Build an executable package:

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
python build.py
```

The executable will be created in the `dist/` directory.

## Security Considerations

### 1. Database Security

- Use strong passwords for database users
- Limit database user privileges to only necessary operations
- Use SSL/TLS for database connections in production
- Regularly update PostgreSQL

### 2. Application Security

- Store configuration files with restricted permissions:
  ```bash
  chmod 600 config/config.prod.json
  ```
- Use environment variables for sensitive data
- Regularly update Python packages

### 3. Network Security

- Use a firewall to restrict access to the application
- Run the application on a dedicated network segment
- Use HTTPS for any web interfaces

## Monitoring and Maintenance

### 1. Log Monitoring

Check application logs regularly:

```bash
# View recent logs
tail -f app.log

# Check for errors
grep -i error app.log
```

### 2. Database Maintenance

Regular database maintenance tasks:

```sql
-- Analyze and vacuum the database
VACUUM ANALYZE;

-- Check for table bloat
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### 3. Backup Strategy

#### Database Backup

```bash
# Create database backup
pg_dump -h localhost -U floor_monitor -d floor_monitor > backup_$(date +%Y%m%d).sql

# Restore database backup
psql -h localhost -U floor_monitor -d floor_monitor < backup_20230101.sql
```

#### Configuration Backup

```bash
# Backup configuration
cp config/config.prod.json config/backup/config.prod.$(date +%Y%m%d).json
```

## Scaling Considerations

### 1. Multiple Cameras

For monitoring multiple areas:

1. Add additional camera configurations in `config.json`
2. Ensure sufficient CPU/GPU resources
3. Consider using multiple application instances for different camera groups

### 2. High Availability

For critical deployments:

1. Set up database replication
2. Use load balancers for multiple application instances
3. Implement automatic failover mechanisms

### 3. Performance Optimization

1. Use GPU acceleration for object detection
2. Optimize camera resolution and frame rates
3. Implement caching for frequently accessed data

## Troubleshooting Production Issues

### 1. Application Won't Start

Check service status:

```bash
# Linux
sudo systemctl status floor-monitor.service

# Windows
sc query FloorMonitor
```

### 2. Database Connection Issues

Verify database connectivity:

```bash
# Test database connection
psql -h your-db-host -U floor_monitor -d floor_monitor
```

### 3. Camera Issues

Check camera access:

```bash
# Linux
lsusb
dmesg | grep -i usb

# Windows
# Check Device Manager for camera devices
```

## Updating the Application

### 1. Backup Current Installation

```bash
# Backup database
pg_dump -h localhost -U floor_monitor -d floor_monitor > backup_before_update.sql

# Backup configuration
cp config/config.prod.json config/backup/config.prod.before_update.json
```

### 2. Update from Repository

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt
```

### 3. Apply Database Migrations

If database schema changes are required:

```sql
-- Apply necessary schema changes
-- Always backup before making schema changes
```

### 4. Restart Services

```bash
# Linux
sudo systemctl restart floor-monitor.service

# Windows
net stop FloorMonitor
net start FloorMonitor
```

## Contact Support

For deployment assistance, contact the development team at [support@floormonitoring.com](mailto:support@floormonitoring.com).