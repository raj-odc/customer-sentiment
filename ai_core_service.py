import re
import yaml
import json
import torch
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline
)
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TRD-defined aspects (10-15 aspects as specified)
PREDEFINED_ASPECTS = [
    "UI/UX", "Performance", "Login", "Battery", "Pricing", 
    "Customer Support", "Features", "Security", "Installation", 
    "Updates", "Compatibility", "Data Usage", "Notifications"
]

class TextPreprocessor:
    """Step 1: Text Preprocessing as per TRD"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and standardize input text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation for context
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        return text

class AspectExtractor:
    """Step 2: Aspect Extraction (AE) - Multi-label Classification"""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize aspect extraction model"""
        self.classifier = pipeline(
            "zero-shot-classification", 
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        self.aspects = PREDEFINED_ASPECTS
        logger.info(f"Aspect Extraction model loaded: {model_name}")
    
    def extract_aspects(self, text: str, threshold: float = 0.3) -> List[str]:
        """
        Extract aspects from text using multi-label classification
        
        Args:
            text: Input text
            threshold: Confidence threshold for aspect detection
            
        Returns:
            List of detected aspect labels
        """
        try:
            if not text.strip():
                return ["General"]
            
            result = self.classifier(text, self.aspects)
            
            # Filter aspects above threshold
            detected_aspects = []
            for label, score in zip(result['labels'], result['scores']):
                if score > threshold:
                    detected_aspects.append(label)
            
            # Return at least one aspect as per TRD requirements
            return detected_aspects if detected_aspects else [result['labels'][0]]
            
        except Exception as e:
            logger.error(f"Error in aspect extraction: {e}")
            return ["General"]

class ABSAModel:
    """Step 3: Aspect-Based Sentiment Analysis (ABSA)"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize ABSA model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Sentiment mapping for continuous score (-1.0 to 1.0)
        self.sentiment_mapping = {
            "LABEL_0": -1.0,  # Negative
            "LABEL_1": 0.0,   # Neutral  
            "LABEL_2": 1.0    # Positive
        }
        logger.info(f"ABSA model loaded: {model_name}")
    
    def analyze_sentiment(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Analyze sentiment for specific aspect in text
        
        Args:
            text: Full sentence
            aspect: Specific aspect to analyze
            
        Returns:
            Dict with sentiment score, label, and confidence
        """
        try:
            # Create aspect-focused input as per TRD methodology
            aspect_text = f"{text} [SEP] {aspect}"
            
            inputs = self.tokenizer(
                aspect_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class_id = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class_id].item()
            
            # Map to sentiment score (-1.0 to 1.0)
            predicted_label = f"LABEL_{predicted_class_id}"
            base_score = self.sentiment_mapping.get(predicted_label, 0.0)
            sentiment_score = base_score * confidence
            
            # Convert to sentiment label
            if sentiment_score > 0.1:
                sentiment_label = "Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            return {
                "sentiment_score": round(sentiment_score, 3),
                "sentiment_label": sentiment_label,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Error in ABSA: {e}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "confidence": 0.0
            }

class KeywordSummaryGenerator:
    """Step 4: Keyword & Summary Generation (for negative topics)"""
    
    def __init__(self):
        """Initialize keyword extraction"""
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def extract_keywords(self, text: str, aspect: str) -> List[str]:
        """Extract key phrases related to the aspect"""
        # Simple keyword extraction based on aspect context
        words = text.lower().split()
        aspect_keywords = []
        
        # Find words around aspect-related terms
        aspect_terms = aspect.lower().split('/')
        for term in aspect_terms:
            for i, word in enumerate(words):
                if term in word:
                    # Get surrounding words
                    start = max(0, i-2)
                    end = min(len(words), i+3)
                    aspect_keywords.extend(words[start:end])
        
        return list(set(aspect_keywords))[:5]  # Return top 5 unique keywords
    
    def generate_summary(self, text: str, max_length: int = 50) -> str:
        """Generate concise summary for negative sentiment"""
        try:
            if len(text) < 50:
                return text
            
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=10, 
                do_sample=False
            )
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return text[:100] + "..." if len(text) > 100 else text

class SuggestionEngine:
    """Step 5: Rule-Based Suggestion Engine"""
    
    def __init__(self, suggestions_file: str = "suggestions.yaml"):
        """Load suggestion rules from YAML file"""
        self.suggestions_file = suggestions_file
        self.rules = self._load_suggestions()
        logger.info(f"Loaded {len(self.rules)} suggestion rules")
    
    def _load_suggestions(self) -> List[Dict]:
        """Load suggestions from YAML file"""
        try:
            with open(self.suggestions_file, 'r') as f:
                return yaml.safe_load(f) or []
        except FileNotFoundError:
            logger.warning(f"Suggestions file {self.suggestions_file} not found")
            return []
        except Exception as e:
            logger.error(f"Error loading suggestions: {e}")
            return []
    
    def get_suggestion(self, aspect: str, sentiment_score: float, text: str, keywords: List[str]) -> Optional[str]:
        """
        Get suggestion based on aspect, sentiment, and keywords
        
        Args:
            aspect: Detected aspect
            sentiment_score: Sentiment score (-1.0 to 1.0)
            text: Original text
            keywords: Extracted keywords
            
        Returns:
            Suggestion string or None
        """
        for rule in self.rules:
            # Check aspect match
            if rule.get("aspect") != aspect:
                continue
            
            # Check sentiment threshold
            sentiment_threshold = rule.get("sentiment_less_than", -1.1)
            if sentiment_score >= sentiment_threshold:
                continue
            
            # Check keyword match
            rule_keywords = rule.get("keywords", [])
            if rule_keywords:
                text_lower = text.lower()
                keyword_match = any(kw.lower() in text_lower for kw in rule_keywords)
                if not keyword_match:
                    continue
            
            return rule.get("suggestion")
        
        return None

class PostgreSQLIntegration:
    """PostgreSQL integration for data storage and retrieval"""
    
    def __init__(self, db_config: Dict[str, str]):
        """Initialize database connection"""
        self.db_config = db_config
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def store_analysis_result(self, text: str, analysis_result: Dict[str, Any]):
        """Store analysis result in PostgreSQL"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ai_analysis_results (
                        original_text, 
                        overall_sentiment, 
                        analysis_json, 
                        processed_at
                    ) VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (
                    text,
                    analysis_result["overall_sentiment"],
                    json.dumps(analysis_result),
                    datetime.now()
                ))
                result_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"Analysis result stored with ID: {result_id}")
                return result_id
        finally:
            conn.close()
    
    def get_similar_analyses(self, text: str, limit: int = 5) -> List[Dict]:
        """Get similar analyses using text similarity (mock implementation)"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Simple similarity based on text length and common words
                cur.execute("""
                    SELECT original_text, analysis_json, processed_at
                    FROM ai_analysis_results
                    WHERE LENGTH(original_text) BETWEEN %s AND %s
                    ORDER BY processed_at DESC
                    LIMIT %s
                """, (len(text) - 50, len(text) + 50, limit))
                return cur.fetchall()
        finally:
            conn.close()

class AICoreService:
    """Main AI Core Service - Orchestrates the entire pipeline"""
    
    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """Initialize all pipeline components"""
        self.preprocessor = TextPreprocessor()
        self.aspect_extractor = AspectExtractor()
        self.absa_model = ABSAModel()
        self.keyword_generator = KeywordSummaryGenerator()
        self.suggestion_engine = SuggestionEngine()
        
        # Optional PostgreSQL integration
        self.db = PostgreSQLIntegration(db_config) if db_config else None
        
        logger.info("AI Core Service initialized successfully")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Main analysis pipeline as per TRD specification
        
        Args:
            text: Raw input text
            
        Returns:
            Structured JSON response matching TRD API contract
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocessing
            cleaned_text = self.preprocessor.clean_text(text)
            if not cleaned_text:
                return self._empty_response(text)
            
            # Step 2: Aspect Extraction
            aspects = self.aspect_extractor.extract_aspects(cleaned_text)
            
            # Step 3-5: Process each aspect
            analysis = []
            overall_sentiment_sum = 0
            overall_sentiment_count = 0
            
            for aspect in aspects:
                # Step 3: ABSA
                sentiment_result = self.absa_model.analyze_sentiment(cleaned_text, aspect)
                
                # Step 4: Keywords & Summary (for negative sentiment)
                keywords = []
                summary = None
                if sentiment_result["sentiment_score"] < -0.3:
                    keywords = self.keyword_generator.extract_keywords(cleaned_text, aspect)
                    summary = self.keyword_generator.generate_summary(cleaned_text)
                
                # Step 5: Suggestion Engine
                suggestion = self.suggestion_engine.get_suggestion(
                    aspect, 
                    sentiment_result["sentiment_score"], 
                    cleaned_text, 
                    keywords
                )
                
                # Find relevant text span
                text_span = self._extract_text_span(text, aspect)
                
                analysis.append({
                    "aspect": aspect,
                    "sentiment_score": sentiment_result["sentiment_score"],
                    "sentiment_label": sentiment_result["sentiment_label"],
                    "text_span": text_span,
                    "suggestion": suggestion,
                    "keywords": keywords if keywords else None,
                    "summary": summary
                })
                
                overall_sentiment_sum += sentiment_result["sentiment_score"]
                overall_sentiment_count += 1
            
            # Calculate overall sentiment
            overall_sentiment = (
                overall_sentiment_sum / overall_sentiment_count 
                if overall_sentiment_count > 0 else 0.0
            )
            
            # Prepare response
            response = {
                "original_text": text,
                "overall_sentiment": round(overall_sentiment, 3),
                "analysis": analysis,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Store in database if available
            if self.db:
                try:
                    self.db.store_analysis_result(text, response)
                except Exception as e:
                    logger.error(f"Database storage failed: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            return self._error_response(text, str(e))
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze multiple texts in batch"""
        results = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch analysis error for text: {text[:50]}... Error: {e}")
                results.append(self._error_response(text, str(e)))
        return results
    
    def _extract_text_span(self, text: str, aspect: str) -> str:
        """Extract relevant text span for the aspect"""
        # Simple implementation - find sentences containing aspect-related words
        sentences = text.split('.')
        aspect_words = aspect.lower().replace('/', ' ').split()
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in aspect_words):
                return sentence.strip()
        
        # Fallback to first 100 characters
        return text[:100] + "..." if len(text) > 100 else text
    
    def _empty_response(self, text: str) -> Dict[str, Any]:
        """Return empty response for invalid input"""
        return {
            "original_text": text,
            "overall_sentiment": 0.0,
            "analysis": [],
            "error": "Empty or invalid text input"
        }
    
    def _error_response(self, text: str, error: str) -> Dict[str, Any]:
        """Return error response"""
        return {
            "original_text": text,
            "overall_sentiment": 0.0,
            "analysis": [],
            "error": error
        }

# Initialize FastAPI app
app = FastAPI(title="Customer Sentiment Analysis MVP", description="Complete sentiment analysis solution")

# Mount static files and templates
templates = Jinja2Templates(directory="templates")

# Initialize AI Core Service with DB config
db_config = {
    "host": "localhost",
    "port": 5432,
    "dbname": "reviews_db",
    "user": "review_user",
    "password": "review123"
}

ai_service = AICoreService(db_config)

@app.post("/analyze")
async def analyze_endpoint(request: Request):
    """
    POST /analyze endpoint as per TRD API contract
    """
    try:
        data = await request.json()
        text = data.get('text', '').strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required and cannot be empty")
        
        # Run analysis synchronously (transformers and DB calls are sync)
        result = ai_service.analyze(text)
        
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        
        return JSONResponse(status_code=200, content=result)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Customer Sentiment Analysis MVP"}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard endpoint"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV or JSON file for batch analysis"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            # Assume the text column is named 'text', 'review', or 'feedback'
            text_column = None
            for col in ['text', 'review', 'feedback', 'comment', 'message']:
                if col in df.columns:
                    text_column = col
                    break
            
            if not text_column:
                raise HTTPException(status_code=400, detail="No valid text column found. Expected columns: text, review, feedback, comment, or message")
            
            results = []
            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()
                if text and text != 'nan':
                    analysis = ai_service.analyze(text)
                    analysis['row_id'] = idx
                    results.append(analysis)
            
            return {"message": f"Processed {len(results)} reviews", "results": results}
            
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            results = []
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    text = item.get('text') or item.get('review') or item.get('feedback')
                    if text:
                        analysis = ai_service.analyze(text)
                        analysis['row_id'] = idx
                        results.append(analysis)
            else:
                text = data.get('text') or data.get('review') or data.get('feedback')
                if text:
                    analysis = ai_service.analyze(text)
                    results.append(analysis)
            
            return {"message": f"Processed {len(results)} reviews", "results": results}
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or JSON files.")
            
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/analytics")
async def get_analytics():
    """Get sentiment analytics from database"""
    try:
        if not ai_service.db:
            return {"error": "Database not configured"}
        
        conn = ai_service.db.get_connection()
        try:
            with conn.cursor() as cur:
                # Get overall sentiment distribution
                cur.execute("""
                    SELECT 
                        CASE 
                            WHEN overall_sentiment > 0.1 THEN 'Positive'
                            WHEN overall_sentiment < -0.1 THEN 'Negative'
                            ELSE 'Neutral'
                        END as sentiment_category,
                        COUNT(*) as count
                    FROM ai_analysis_results
                    GROUP BY sentiment_category
                """)
                sentiment_distribution = dict(cur.fetchall())
                
                # Get recent analyses
                cur.execute("""
                    SELECT original_text, overall_sentiment, processed_at
                    FROM ai_analysis_results
                    ORDER BY processed_at DESC
                    LIMIT 10
                """)
                recent_analyses = cur.fetchall()
                
                # Get average sentiment over time (last 30 days)
                cur.execute("""
                    SELECT 
                        DATE(processed_at) as date,
                        AVG(overall_sentiment) as avg_sentiment,
                        COUNT(*) as count
                    FROM ai_analysis_results
                    WHERE processed_at >= NOW() - INTERVAL '30 days'
                    GROUP BY DATE(processed_at)
                    ORDER BY date
                """)
                sentiment_trends = cur.fetchall()
                
                return {
                    "sentiment_distribution": sentiment_distribution,
                    "recent_analyses": [
                        {
                            "text": row[0][:100] + "..." if len(row[0]) > 100 else row[0],
                            "sentiment": row[1],
                            "processed_at": row[2].isoformat()
                        } for row in recent_analyses
                    ],
                    "sentiment_trends": [
                        {
                            "date": row[0].isoformat(),
                            "avg_sentiment": float(row[1]),
                            "count": row[2]
                        } for row in sentiment_trends
                    ]
                }
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {"error": f"Failed to fetch analytics: {str(e)}"}

@app.get("/export")
async def export_results():
    """Export all analysis results as JSON"""
    try:
        if not ai_service.db:
            return {"error": "Database not configured"}
        
        conn = ai_service.db.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT original_text, overall_sentiment, analysis_json, processed_at
                    FROM ai_analysis_results
                    ORDER BY processed_at DESC
                """)
                results = cur.fetchall()
                
                export_data = []
                for row in results:
                    export_data.append({
                        "original_text": row[0],
                        "overall_sentiment": row[1],
                        "analysis": json.loads(row[2]) if row[2] else {},
                        "processed_at": row[3].isoformat()
                    })
                
                return {"data": export_data, "count": len(export_data)}
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Export error: {e}")
        return {"error": f"Failed to export results: {str(e)}"}

# Simple sentiment analysis fallback for lightweight usage
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@app.post("/analyze/simple")
async def analyze_simple(request: Request):
    """Simple sentiment analysis using TextBlob and VADER (faster, lighter)"""
    try:
        data = await request.json()
        text = data.get('text', '').strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity  # -1 to 1
        
        # VADER analysis
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        vader_sentiment = vader_scores['compound']  # -1 to 1
        
        # Combined score
        combined_sentiment = (textblob_sentiment + vader_sentiment) / 2
        
        # Determine label
        if combined_sentiment > 0.1:
            sentiment_label = "Positive"
        elif combined_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        result = {
            "original_text": text,
            "sentiment_score": round(combined_sentiment, 3),
            "sentiment_label": sentiment_label,
            "textblob_score": round(textblob_sentiment, 3),
            "vader_score": round(vader_sentiment, 3),
            "confidence": round(abs(combined_sentiment), 3)
        }
        
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        logger.error(f"Simple analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# To run:
# uvicorn ai_core_service:app --host 0.0.0.0 --port 5000 --reload