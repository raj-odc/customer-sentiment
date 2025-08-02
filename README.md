# Customer Sentiment Analysis MVP

A complete end-to-end solution for analyzing customer sentiment from feedback, reviews, and text data. This MVP combines advanced AI models with a user-friendly interface to provide actionable insights from customer feedback.

## 🚀 Features

### Core Functionality
- **Advanced Sentiment Analysis**: Uses transformer models (RoBERTa) for accurate sentiment detection
- **Aspect-Based Analysis**: Identifies specific aspects (UI/UX, Performance, Login, etc.) and their sentiment
- **Multi-Model Support**: Includes both advanced AI models and lightweight alternatives (TextBlob, VADER)
- **Batch Processing**: Upload CSV/JSON files for bulk analysis
- **Real-time Dashboard**: Interactive web interface for viewing results

### Data Processing
- **Multiple Input Formats**: Supports CSV, JSON, and direct text input
- **Google Reviews Integration**: Scrape and analyze Google Reviews using SerpAPI
- **Database Storage**: PostgreSQL backend with full data persistence
- **Export Functionality**: Download analysis results as JSON

### Analytics & Insights
- **Sentiment Distribution**: Visual breakdown of positive/negative/neutral feedback
- **Trend Analysis**: Track sentiment changes over time
- **Suggestion Engine**: Rule-based recommendations for addressing negative feedback
- **Keyword Extraction**: Identify key themes in negative feedback

## 📊 Demo

Access the live dashboard at: `http://localhost:5000`

### Quick Start Example
```bash
# 1. Setup database
python start_mvp.py --mode setup

# 2. Start development server
python start_mvp.py --mode dev

# 3. Test the system
curl -X POST "http://localhost:5000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "The app is great but login is slow and crashes sometimes"}'
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Git

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd customer-sentiment

# Install dependencies
pip install -r requirements.txt

# Setup database and start
python start_mvp.py --mode setup
python start_mvp.py --mode dev
```

### Docker Setup (Recommended)
```bash
# Start with Docker Compose
python start_mvp.py --mode docker

# Or manually
docker-compose up --build
```

## 📁 Project Structure

```
customer-sentiment/
├── ai_core_service.py      # Main FastAPI application
├── setup_database.py       # Database initialization
├── start_mvp.py            # Startup script
├── requirements.txt        # Python dependencies
├── suggestions.yaml        # Rule-based suggestions
├── templates/
│   └── dashboard.html      # Web dashboard
├── sample_data.csv         # Test data (CSV format)
├── sample_data.json        # Test data (JSON format)
├── DB/                     # Database backups
├── docker-compose.yml      # Docker configuration
└── Dockerfile             # Docker image definition
```

## 🔧 API Endpoints

### Core Analysis
- `POST /analyze` - Complete sentiment analysis with aspects
- `POST /analyze/simple` - Quick sentiment analysis (TextBlob + VADER)
- `POST /upload` - Batch file upload (CSV/JSON)

### Data & Analytics
- `GET /analytics` - Get sentiment statistics and trends
- `GET /export` - Export all analysis results
- `GET /health` - Health check endpoint

### Dashboard
- `GET /` - Interactive web dashboard

## 📊 Usage Examples

### Single Text Analysis
```python
import requests

response = requests.post('http://localhost:5000/analyze', json={
    "text": "The UI is beautiful but the app crashes on startup"
})

result = response.json()
print(f"Overall sentiment: {result['overall_sentiment']}")
for aspect in result['analysis']:
    print(f"{aspect['aspect']}: {aspect['sentiment_label']}")
```

### Batch File Upload
```python
files = {'file': open('customer_feedback.csv', 'rb')}
response = requests.post('http://localhost:5000/upload', files=files)
results = response.json()
```

### CSV Format
```csv
text,source,rating
"Great app, love the features!",app_store,5
"Crashes frequently, needs fixes",app_store,2
```

### JSON Format
```json
[
  {"text": "Excellent service!", "rating": 5},
  {"text": "Poor quality, disappointed", "rating": 1}
]
```

## 🎯 Sentiment Analysis Pipeline

### 1. Text Preprocessing
- Cleaning and normalization
- Special character handling
- Text standardization

### 2. Aspect Extraction
- Multi-label classification using BART
- 13 predefined aspects: UI/UX, Performance, Login, Battery, etc.
- Configurable confidence thresholds

### 3. Aspect-Based Sentiment Analysis (ABSA)
- RoBERTa-based sentiment classification
- Aspect-specific sentiment scoring (-1.0 to +1.0)
- Confidence scoring

### 4. Keyword & Summary Generation
- Automatic keyword extraction for negative feedback
- Text summarization for long reviews
- Context-aware phrase identification

### 5. Rule-Based Suggestions
- YAML-configured suggestion rules
- Context-aware recommendations
- Actionable insights for improvement

## 📈 Analytics Dashboard

The web dashboard provides:

- **Real-time Sentiment Distribution**: Pie chart showing positive/negative/neutral breakdown
- **Trend Analysis**: Line chart tracking sentiment over time
- **Recent Analyses**: List of latest processed feedback
- **Batch Upload**: Drag-and-drop file processing
- **Export Options**: Download results in JSON format

## 🔧 Configuration

### Database Configuration
Edit database settings in `ai_core_service.py`:
```python
db_config = {
    "host": "localhost",
    "port": 5432,
    "dbname": "reviews_db",
    "user": "review_user",
    "password": "review123"
}
```

### Suggestion Rules
Customize suggestions in `suggestions.yaml`:
```yaml
- rule_name: Login_Issues
  aspect: Login
  sentiment_less_than: -0.5
  keywords: ["slow", "timeout", "failed"]
  suggestion: "Optimize login API performance and add timeout handling"
```

## 🚀 Deployment Options

### Development
```bash
python start_mvp.py --mode dev
```

### Production
```bash
python start_mvp.py --mode prod
```

### Docker
```bash
docker-compose up -d
```

### Cloud Deployment
The application is ready for deployment on:
- Heroku
- AWS (ECS/EKS)
- Google Cloud Run
- Azure Container Instances

## 🧪 Testing

### Run Tests
```bash
python start_mvp.py --mode test
```

### Manual Testing
```bash
# Test basic analysis
python test_sentiment_analysis.py

# Test batch processing
python batch_sentiment_analysis.py
```

## 📊 Performance

### Processing Speed
- **Simple Analysis**: ~100ms per text
- **Advanced Analysis**: ~2-5 seconds per text (first run includes model loading)
- **Batch Processing**: Parallel processing support

### Model Sizes
- **RoBERTa Sentiment**: ~500MB
- **BART Classification**: ~1.6GB
- **BART Summarization**: ~1.6GB

### Hardware Requirements
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 5GB disk space
- **GPU**: Optional, speeds up processing by 3-5x

## 🔒 Security & Privacy

### Data Protection
- All data stored locally in PostgreSQL
- No external API calls for sentiment analysis
- Optional cloud model downloads only

### API Security
- Input validation and sanitization
- Error handling and logging
- Rate limiting ready (can be enabled)

## 🤝 Contributing

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run with hot reload
python start_mvp.py --mode dev

# Format code
black *.py

# Type checking
mypy ai_core_service.py
```

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📋 Roadmap

### Version 1.1 (Next Release)
- [ ] Real-time streaming analysis
- [ ] Custom model training
- [ ] Advanced visualization options
- [ ] Email/Slack notifications

### Version 2.0 (Future)
- [ ] Multi-language support
- [ ] Voice feedback analysis
- [ ] Integration with popular platforms (Zendesk, Intercom)
- [ ] Machine learning model retraining

## 🐛 Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL is running
sudo service postgresql status

# Reset database
python start_mvp.py --mode setup
```

**Model Download Issues**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python start_mvp.py --mode dev
```

**Port Already in Use**
```bash
# Kill process on port 5000
sudo lsof -ti:5000 | xargs kill -9
```

### Performance Issues
- Reduce model size by using simple analysis mode
- Add more RAM or use GPU acceleration
- Enable batch processing for multiple texts

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting guide
- Review the API documentation at `/docs`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for rapid prototyping and customer insights**