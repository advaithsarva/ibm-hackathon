# Disaster Early Warning Network

A global early-warning network where no disaster catches humanity unprepared and every vulnerable person receives help before catastrophe strikes through unified predictive intelligence.

## Product Vision

Create a unified platform that enables emergency operations directors, first responder teams, municipal emergency planners, and vulnerable community populations to receive timely disaster alerts and coordinate response efforts effectively.

## Target Audience

- Emergency operations directors
- First responder teams
- Municipal emergency planners
- Vulnerable community populations requiring priority assistance

## Core Features

- **CRUD Operations for Disaster Alerts**: Create, read, update, and delete disaster alerts
- **Disaster Type Classification**: Support for earthquakes, floods, hurricanes, wildfires, tsunamis, tornadoes, droughts, and volcanic events
- **Severity Levels**: Track disasters by severity (low, moderate, high, critical)
- **Geographic Tracking**: Location-based disaster monitoring with latitude/longitude coordinates
- **Impact Assessment**: Track affected population and predicted impact

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: SQLite (SQLAlchemy ORM)
- **Architecture**: Modular Monolith

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

1. Clone the repository or navigate to the project directory

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

4. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

5. Create environment configuration:
```bash
cp ../.env.example .env
```

6. Edit `.env` file with your configuration (optional for local development)

## Running Locally

1. Make sure you're in the backend directory with the virtual environment activated

2. Start the development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. The API will be available at:
   - API: http://localhost:8000
   - Interactive API docs: http://localhost:8000/docs
   - Alternative API docs: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health status

### Disaster Management
- `POST /api/v1/disasters/` - Create a new disaster alert
- `GET /api/v1/disasters/` - List all disasters (with optional filters)
- `GET /api/v1/disasters/{disaster_id}` - Get specific disaster details
- `PUT /api/v1/disasters/{disaster_id}` - Update disaster information
- `DELETE /api/v1/disasters/{disaster_id}` - Delete a disaster alert

### Query Parameters for Listing
- `skip` - Number of records to skip (pagination)
- `limit` - Maximum number of records to return
- `disaster_type` - Filter by disaster type
- `severity` - Filter by severity level
- `status` - Filter by status

## Example API Usage

### Create a Disaster Alert
```bash
curl -X POST "http://localhost:8000/api/v1/disasters/" \
  -H "Content-Type: application/json" \
  -d '{
    "disaster_type": "earthquake",
    "severity": "high",
    "location": "San Francisco, CA",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "description": "Magnitude 6.5 earthquake detected",
    "predicted_impact": "Significant structural damage expected",
    "affected_population": 50000
  }'
```

### List All Disasters
```bash
curl "http://localhost:8000/api/v1/disasters/"
```

### Get Specific Disaster
```bash
curl "http://localhost:8000/api/v1/disasters/1"
```

### Update Disaster
```bash
curl -X PUT "http://localhost:8000/api/v1/disasters/1" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "resolved",
    "affected_population": 75000
  }'
```

### Delete Disaster
```bash
curl -X DELETE "http://localhost:8000/api/v1/disasters/1"
```

## Database Schema

### Disaster Model
- `id` - Primary key
- `disaster_type` - Type of disaster (enum)
- `severity` - Severity level (enum)
- `location` - Human-readable location
- `latitude` - Geographic latitude
- `longitude` - Geographic longitude
- `description` - Detailed description
- `predicted_impact` - Expected impact assessment
- `affected_population` - Number of people affected
- `status` - Current status (default: "active")
- `detected_at` - When the disaster was detected
- `created_at` - Record creation timestamp
- `updated_at` - Last update timestamp

## Architecture Overview

The application follows a **Modular Monolith** architecture with clear separation of concerns:

- `main.py` - Application entry point and configuration
- `models.py` - Database models and ORM setup
- `routers/` - API route handlers organized by domain
- `config.py` - Configuration management

## Development

The application uses:
- **FastAPI** for high-performance async API
- **SQLAlchemy** for database ORM
- **Pydantic** for data validation
- **Uvicorn** as ASGI server

## Security Considerations

- Input validation using Pydantic models
- SQL injection prevention through SQLAlchemy ORM
- CORS configuration for cross-origin requests
- Environment variables for sensitive configuration

## Future Enhancements

- Authentication and authorization
- Real-time notifications via WebSockets
- Integration with external disaster monitoring APIs
- Machine learning for disaster prediction
- Mobile application support
- Multi-language support
