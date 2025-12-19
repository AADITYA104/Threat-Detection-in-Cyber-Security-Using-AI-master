# CyberShield AI - Frontend

Modern, elegant web interface for the AI-powered threat detection system.

## Features

- **Modern Design**: Glassmorphism UI with dark theme
- **Real-time Detection**: Upload and analyze network traffic data
- **Model Training**: Train custom ML models with visual feedback
- **Interactive Charts**: Beautiful visualizations using Chart.js
- **Responsive**: Works seamlessly on desktop and mobile devices
- **Smooth Animations**: Engaging micro-interactions and transitions

## Technologies Used

- HTML5
- CSS3 (Custom glassmorphism design)
- Vanilla JavaScript
- Chart.js for data visualization
- Google Fonts (Inter)

## Getting Started

1. Ensure the backend API is running on `http://localhost:5000`

2. Open `index.html` in your browser:
   - Simply double-click the file, or
   - Use a local server (recommended):
     ```bash
     # If you have Python installed
     python -m http.server 8000
     ```
   Then navigate to `http://localhost:8000`

## Features Overview

### Dashboard
- System status monitoring
- Quick action buttons
- Real-time statistics

### Threat Detection
- Drag-and-drop file upload
- CSV file analysis
- Real-time threat classification

### Model Training
- Multiple algorithm support
- Feature selection options
- Training metrics visualization
- Model management

### Statistics
- Attack distribution charts
- Threat percentage breakdown
- Detailed attack type cards

## Design System

The application uses a modern design system with:
- **Dark Theme**: Optimized for extended use
- **Glassmorphism**: Translucent cards with backdrop blur
- **Gradient Accents**: Vibrant purple and blue gradients
- **Smooth Transitions**: 0.3s ease transitions throughout
- **Responsive Grid**: Adapts to all screen sizes

## API Configuration

By default, the frontend connects to `http://localhost:5000/api`

To change this, edit `app.js`:
```javascript
const API_BASE_URL = 'http://your-backend-url/api';
```

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

Modern browsers with ES6+ support required.
