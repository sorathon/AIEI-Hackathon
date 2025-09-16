import pandas as pd
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import json
import ast
import logging
from typing import List, Dict, Tuple, Any
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QwenHospitalRecommendationBot:
    def __init__(self, config_file: str = "C:\\Users\\mattm\\Downloads\\line_config.json"):
        """
        Initialize the Qwen Hospital Recommendation Bot
        
        Args:
            config_file: Path to Line API configuration file
        """
        self.load_config(config_file)
        self.load_qwen_model()
        self.load_hospital_data()
        
        # Initialize Line Bot API
        self.line_bot_api = LineBotApi(self.channel_access_token)
        self.handler = WebhookHandler(self.channel_secret)
        
        # Hospital preference settings
        self.preferred_hospitals = [
            "ศูนย์สุขภาพรามาธิบดี พาราไดซ์พาร์ค",
            "ศูนย์สุขภาพรามาธิบดี โลตัส นอร์ธ ราชพฤกษ์"
        ]
        
        self.comparison_hospitals = [
            "คณะแพทยศาสตร์โรงพยาบาลรามาธิบดี มหาวิทยาลัยมหิดล",
            "สถาบันการแพทย์จักรีนฤบดินทร์ คณะแพทยศาสตร์โรงพยาบาลรามาธิบดี มหาวิทยาลัยมหิดล"
        ]
        
        # Initialize recommendation history tracking
        self.history_file = "recommendation_history.json"
        self.recommendation_history = self.load_recommendation_history()
        
    def load_config(self, config_file: str):
        """Load Line API configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.channel_access_token = config['channel_access_token']
                self.channel_secret = config['channel_secret']
            else:
                # Use environment variables as fallback
                self.channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
                self.channel_secret = os.getenv('LINE_CHANNEL_SECRET')
                
            if not self.channel_access_token or not self.channel_secret:
                raise ValueError("Line API credentials not found")
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def load_recommendation_history(self) -> Dict:
        """Load recommendation history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"Loaded recommendation history: {len(history)} records")
                return history
            else:
                logger.info("No existing recommendation history found, starting fresh")
                return {}
        except Exception as e:
            logger.error(f"Error loading recommendation history: {e}")
            return {}
    
    def save_recommendation_history(self):
        """Save recommendation history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.recommendation_history, f, ensure_ascii=False, indent=2)
            logger.info("Recommendation history saved successfully")
        except Exception as e:
            logger.error(f"Error saving recommendation history: {e}")
    
    def update_recommendation_history(self, hospital_name: str, user_id: str = "general"):
        """Update recommendation history for a user"""
        current_time = datetime.now().isoformat()
        
        if user_id not in self.recommendation_history:
            self.recommendation_history[user_id] = []
        
        # Add new recommendation
        self.recommendation_history[user_id].append({
            'hospital': hospital_name,
            'timestamp': current_time
        })
        
        # Keep only last 10 recommendations per user to prevent file bloat
        self.recommendation_history[user_id] = self.recommendation_history[user_id][-10:]
        
        # Save to file
        self.save_recommendation_history()
        
        logger.info(f"Updated recommendation history for user {user_id}: {hospital_name}")
    
    def calculate_history_penalty(self, hospital_name: str, user_id: str = "general") -> float:
        """
        Calculate penalty score based on recent recommendation history
        Returns a penalty factor (0.0 to 1.0) where lower means more penalty
        """
        if user_id not in self.recommendation_history:
            return 1.0  # No penalty for first-time users
        
        user_history = self.recommendation_history[user_id]
        if len(user_history) < 2:
            return 1.0  # No penalty if less than 2 recommendations
        
        # Check last 2 recommendations
        recent_recommendations = user_history[-2:]
        recent_hospitals = [rec['hospital'] for rec in recent_recommendations]
        
        # Count consecutive occurrences of this hospital
        consecutive_count = 0
        for i in range(len(recent_hospitals) - 1, -1, -1):
            if recent_hospitals[i] == hospital_name:
                consecutive_count += 1
            else:
                break
        
        # Apply penalty based on consecutive recommendations
        if consecutive_count >= 2:
            return 0.1  # Strong penalty for 2+ consecutive recommendations
        elif consecutive_count == 1:
            return 0.7  # Moderate penalty for 1 recent recommendation
        else:
            return 1.0  # No penalty
    
    def load_qwen_model(self):
        """Load the Qwen3 Reranker model for embeddings"""
        try:
            model_name = "Qwen/Qwen3-Reranker-0.6B"
            logger.info(f"Loading Qwen model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.float32,
                trust_remote_code=True
            )
            
            # Move to CPU and set to evaluation mode
            self.model = self.model.to('cpu')
            self.model.eval()
            
            logger.info("Qwen model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Qwen model: {e}")
            raise
    
    def get_qwen_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using Qwen3 Reranker model
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array representing the embedding
        """
        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.squeeze().cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text}': {e}")
            return np.array([])
    
    def load_hospital_data(self):
        """Load hospital disease records with embeddings"""
        try:
            self.hospital_df = pd.read_csv('hospital_dis_records_with_actual_embeddings.csv', encoding='utf-8')
            logger.info(f"Loaded {len(self.hospital_df)} hospital records")
            
            # Parse embeddings from string format
            self.parse_embeddings()
            
        except Exception as e:
            logger.error(f"Error loading hospital data: {e}")
            raise
    
    def parse_embeddings(self):
        """Parse embedding strings back to numpy arrays"""
        try:
            # Parse disease name embeddings
            disease_embeddings = []
            for idx, embedding_str in enumerate(self.hospital_df['Disease Name Embedding']):
                if isinstance(embedding_str, str) and embedding_str.startswith('['):
                    try:
                        embedding = ast.literal_eval(embedding_str)
                        disease_embeddings.append(np.array(embedding))
                    except:
                        # Generate new embedding if parsing fails
                        disease_name = self.hospital_df.iloc[idx]['Disease Name']
                        logger.warning(f"Re-generating embedding for disease: {disease_name}")
                        embedding = self.get_qwen_embedding(disease_name)
                        disease_embeddings.append(embedding)
                else:
                    # Generate embedding if not available
                    disease_name = self.hospital_df.iloc[idx]['Disease Name']
                    embedding = self.get_qwen_embedding(disease_name)
                    disease_embeddings.append(embedding)
            
            self.hospital_df['parsed_disease_embedding'] = disease_embeddings
            
            # Parse symptom embeddings
            symptom_embeddings = []
            for symptoms_str in self.hospital_df['Disease Symptom List']:
                symptoms_data = self.parse_symptoms_with_embeddings(symptoms_str)
                symptom_embeddings.append(symptoms_data)
            
            self.hospital_df['parsed_symptom_embeddings'] = symptom_embeddings
            
        except Exception as e:
            logger.error(f"Error parsing embeddings: {e}")
            raise
    
    def parse_symptoms_with_embeddings(self, symptoms_str: str) -> List[Dict]:
        """Parse symptoms string with embeddings using Qwen model"""
        symptoms_data = []
        
        # Remove quotes and split by semicolon or comma
        symptoms_str = symptoms_str.strip('"\'')
        
        # Handle different formats from the CSV
        if ';' in symptoms_str:
            symptoms_list = symptoms_str.split(';')
        else:
            # Split by comma, but be careful with embedded arrays
            symptoms_list = []
            current_symptom = ""
            bracket_count = 0
            
            for char in symptoms_str:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == ',' and bracket_count == 0:
                    if current_symptom.strip():
                        symptoms_list.append(current_symptom.strip())
                    current_symptom = ""
                    continue
                
                current_symptom += char
            
            # Add the last symptom
            if current_symptom.strip():
                symptoms_list.append(current_symptom.strip())
        
        for symptom_item in symptoms_list:
            symptom_item = symptom_item.strip()
            
            # Try to extract symptom text and embedding
            if '[' in symptom_item and ']' in symptom_item:
                # Format: "symptom,[embedding]" or "symptom text,[array]"
                bracket_start = symptom_item.find('[')
                symptom_text = symptom_item[:bracket_start].strip()
                embedding_str = symptom_item[bracket_start:]
                
                try:
                    embedding = ast.literal_eval(embedding_str)
                    symptoms_data.append({
                        'text': symptom_text,
                        'embedding': np.array(embedding)
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse embedding for '{symptom_text}', regenerating: {e}")
                    # Generate new embedding using Qwen
                    embedding = self.get_qwen_embedding(symptom_text)
                    symptoms_data.append({
                        'text': symptom_text,
                        'embedding': embedding
                    })
            else:
                # Plain text symptom - generate embedding using Qwen
                if symptom_item:  # Only if not empty
                    embedding = self.get_qwen_embedding(symptom_item)
                    symptoms_data.append({
                        'text': symptom_item,
                        'embedding': embedding
                    })
        
        return symptoms_data
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into individual words"""
        # Remove punctuation and split into words
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [word.strip().lower() for word in words if word.strip()]
    
    def calculate_text_match(self, user_words: List[str], symptom_words: List[str]) -> float:
        """Calculate enhanced text matching score with exact word match bonus"""
        if not user_words or not symptom_words:
            return 0.0
        
        exact_matches = 0
        partial_matches = 0
        total_words = len(user_words)
        
        # Convert symptom words to lowercase for comparison
        symptom_words_lower = [word.lower() for word in symptom_words]
        
        for user_word in user_words:
            user_word_lower = user_word.lower()
            
            # Check for exact matches (highest priority)
            if user_word_lower in symptom_words_lower:
                exact_matches += 1
            # Check for partial matches (substring matching)
            else:
                for symptom_word in symptom_words_lower:
                    if (user_word_lower in symptom_word or symptom_word in user_word_lower) and len(user_word_lower) > 2:
                        partial_matches += 1
                        break
        
        # Calculate weighted score: exact matches get full weight, partial matches get 0.5 weight
        weighted_matches = exact_matches + (partial_matches * 0.5)
        match_score = weighted_matches / total_words if total_words > 0 else 0.0
        
        # Bonus for high exact match ratio
        exact_match_ratio = exact_matches / total_words if total_words > 0 else 0.0
        if exact_match_ratio > 0.5:  # If more than 50% are exact matches
            match_score *= 1.2  # 20% bonus
        
        return min(match_score, 1.0)  # Cap at 1.0
    
    def calculate_embedding_similarity(self, user_embedding: np.ndarray, symptom_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Ensure embeddings are not empty
            if user_embedding.size == 0 or symptom_embedding.size == 0:
                return 0.0
            
            # Reshape to 2D arrays for cosine_similarity
            user_emb_2d = user_embedding.reshape(1, -1)
            symptom_emb_2d = symptom_embedding.reshape(1, -1)
            
            similarity = cosine_similarity(user_emb_2d, symptom_emb_2d)[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def calculate_disease_similarity(self, user_text: str) -> List[Dict]:
        """Calculate similarity scores for all diseases with enhanced word matching"""
        user_embedding = self.get_qwen_embedding(user_text)
        user_words = self.preprocess_text(user_text)
        
        disease_scores = []
        
        for idx, row in self.hospital_df.iterrows():
            disease_name = row['Disease Name']
            hospital_name = row['สถานที่']
            symptom_embeddings = row['parsed_symptom_embeddings']
            
            # Calculate similarity with each symptom
            symptom_scores = []
            
            for symptom_data in symptom_embeddings:
                symptom_text = symptom_data['text']
                symptom_embedding = symptom_data['embedding']
                symptom_words = self.preprocess_text(symptom_text)
                
                # Enhanced text matching score with exact word bonus
                text_match = self.calculate_text_match(user_words, symptom_words)
                
                # Embedding similarity score
                embedding_similarity = self.calculate_embedding_similarity(user_embedding, symptom_embedding)
                
                # Modified combined score: Give more weight to text matching
                # New formula: (3 * text_match + 2 * embedding_similarity) / 5
                # This gives 60% weight to text matching and 40% to embedding similarity
                combined_score = (3 * text_match + 2 * embedding_similarity) / 5
                symptom_scores.append(combined_score)
            
            # Average symptom scores for this disease
            avg_symptom_score = np.mean(symptom_scores) if symptom_scores else 0.0
            
            disease_scores.append({
                'hospital': hospital_name,
                'disease': disease_name,
                'similarity_score': avg_symptom_score,
                'patient_count': row['patient_record'],
                'google_map': row['Google Map']
            })
        
        # Sort by similarity score (descending)
        disease_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return disease_scores
    
    def calculate_hospital_scores(self, ranked_diseases: List[Dict]) -> List[Dict]:
        """Calculate hospital specialization and recommendation scores"""
        # Get top 5 diseases
        top_5_diseases = [disease['disease'] for disease in ranked_diseases[:5]]
        
        # Group by hospital
        hospital_stats = {}
        
        for _, row in self.hospital_df.iterrows():
            hospital = row['สถานที่']
            disease = row['Disease Name']
            patient_count = row['patient_record']
            
            if hospital not in hospital_stats:
                hospital_stats[hospital] = {
                    'total_patients': 0,
                    'top5_patients': 0,
                    'diseases': []
                }
            
            hospital_stats[hospital]['total_patients'] += patient_count
            hospital_stats[hospital]['diseases'].append(disease)
            
            if disease in top_5_diseases:
                hospital_stats[hospital]['top5_patients'] += patient_count
        
        # Calculate specialization scores
        hospital_scores = []
        specializations = []
        
        for hospital, stats in hospital_stats.items():
            if stats['total_patients'] > 0:
                specialization = stats['top5_patients'] / stats['total_patients']
                specializations.append(specialization)
                
                hospital_scores.append({
                    'hospital': hospital,
                    'specialization': specialization,
                    'total_patients': stats['total_patients'],
                    'top5_patients': stats['top5_patients']
                })
        
        # Calculate hospital scores
        for hospital_data in hospital_scores:
            hospital_data['hospital_score'] = hospital_data['specialization']
        
        # Sort by hospital score (descending)
        hospital_scores.sort(key=lambda x: x['hospital_score'], reverse=True)
        
        return hospital_scores
    
    def get_hospital_recommendation(self, hospital_scores: List[Dict], user_id: str = "general") -> Dict:
        """Apply recommendation logic with history consideration"""
        # Apply history penalty to all hospitals
        for hospital_data in hospital_scores:
            hospital = hospital_data['hospital']
            history_penalty = self.calculate_history_penalty(hospital, user_id)
            
            # Apply penalty to the hospital score
            original_score = hospital_data['hospital_score']
            hospital_data['hospital_score_with_history'] = original_score * history_penalty
            hospital_data['history_penalty'] = history_penalty
            
            logger.info(f"Hospital: {hospital}, Original Score: {original_score:.3f}, "
                       f"History Penalty: {history_penalty:.3f}, "
                       f"Final Score: {hospital_data['hospital_score_with_history']:.3f}")
        
        # Re-sort by score with history penalty
        hospital_scores.sort(key=lambda x: x['hospital_score_with_history'], reverse=True)
        
        # Find scores for preferred and comparison hospitals
        preferred_scores = {}
        comparison_scores = {}
        
        for hospital_data in hospital_scores:
            hospital = hospital_data['hospital']
            score = hospital_data['hospital_score_with_history']
            
            if hospital in self.preferred_hospitals:
                preferred_scores[hospital] = hospital_data
            elif hospital in self.comparison_hospitals:
                comparison_scores[hospital] = hospital_data
        
        # Get highest scoring comparison hospital
        if comparison_scores:
            best_comparison = max(comparison_scores.values(), key=lambda x: x['hospital_score_with_history'])
            best_comparison_score = best_comparison['hospital_score_with_history']
        else:
            best_comparison_score = 0
        
        # Check if any preferred hospital is within 0.1 points
        recommended_hospital = None
        
        for hospital in self.preferred_hospitals:
            if hospital in preferred_scores:
                preferred_score = preferred_scores[hospital]['hospital_score_with_history']
                if preferred_score >= (best_comparison_score - 0.1):
                    if recommended_hospital is None or preferred_score > preferred_scores[recommended_hospital]['hospital_score_with_history']:
                        recommended_hospital = hospital
        
        if recommended_hospital:
            final_recommendation = preferred_scores[recommended_hospital]
        else:
            # Return highest scoring hospital overall
            final_recommendation = hospital_scores[0] if hospital_scores else {}
        
        # Update recommendation history
        if final_recommendation:
            self.update_recommendation_history(final_recommendation['hospital'], user_id)
        
        return final_recommendation
    
    def save_results_to_csv(self, ranked_diseases: List[Dict], hospital_scores: List[Dict], user_text: str):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save disease rankings
        disease_df = pd.DataFrame(ranked_diseases)
        disease_filename = f"disease_order_{timestamp}.csv"
        disease_df.to_csv(disease_filename, index=False, encoding='utf-8')
        
        # Save hospital scores
        hospital_df = pd.DataFrame(hospital_scores)
        hospital_filename = f"hospitals_recommendation_score_{timestamp}.csv"
        hospital_df.to_csv(hospital_filename, index=False, encoding='utf-8')
        
        logger.info(f"Results saved: {disease_filename}, {hospital_filename}")
        
        return disease_filename, hospital_filename
    
    def process_user_symptoms(self, user_text: str, user_id: str = "general") -> str:
        """Main processing function for user symptoms"""
        try:
            logger.info(f"Processing user input: {user_text} (User ID: {user_id})")
            
            # Calculate disease similarities
            ranked_diseases = self.calculate_disease_similarity(user_text)
            
            # Calculate hospital scores
            hospital_scores = self.calculate_hospital_scores(ranked_diseases)
            
            # Get recommendation with history consideration
            recommended_hospital = self.get_hospital_recommendation(hospital_scores, user_id)
            
            # Save results to CSV
            self.save_results_to_csv(ranked_diseases, hospital_scores, user_text)
            
            # Format response message
            response_message = self.format_response_message(recommended_hospital, ranked_diseases[:3])
            
            return response_message
            
        except Exception as e:
            logger.error(f"Error processing symptoms: {e}")
            return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"
    
    def format_response_message(self, recommended_hospital: Dict, top_diseases: List[Dict]) -> str:
        """Format the response message for Line"""
        if not recommended_hospital:
            return "ขออภัย ไม่พบข้อมูลโรงพยาบาลที่เหมาะสม"
        
        hospital_name = recommended_hospital['hospital']
        hospital_score = recommended_hospital.get('hospital_score_with_history', recommended_hospital['hospital_score'])
        hospital_sco = hospital_score/2 + 0.5
        
        message = f"🏥 โรงพยาบาลที่แนะนำ:\n{hospital_name}\n"
        message += f"📊 คะแนนความเชี่ยวชาญ: {hospital_sco:.2f}\n"
        
        # Show history penalty info if applied
        if 'history_penalty' in recommended_hospital and recommended_hospital['history_penalty'] < 1.0:
            message += f"📝 คำนึงถึงประวัติการแนะนำก่อนหน้า\n"
        
        message += "\n"
        
        #message += "🔍 โรคที่น่าจะเป็นไปได้:\n"
        for i, disease in enumerate(top_diseases[:1], 1):
            similarity = disease['similarity_score'] * 100
            message += f"🔍 โรคที่น่าจะเป็นไปได้:\n"
            message += f" {disease['disease']} ({similarity:.1f}%)\n"
        
        # Add Google Maps link if available
        hospital_map = None
        for disease in top_diseases:
            if disease['hospital'] == hospital_name:
                hospital_map = disease.get('google_map')
                break
        
        if hospital_map:
            message += f"\n📍 แผนที่: {hospital_map}"
        
        message += "\n\n⚠️ ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"
        
        return message

# Initialize bot instance
bot = QwenHospitalRecommendationBot()

@app.route("/callback", methods=['POST'])
def callback():
    """Line webhook callback"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        bot.handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    
    return 'OK'

@bot.handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """Handle incoming text messages"""
    user_text = event.message.text
    user_id = event.source.user_id if hasattr(event.source, 'user_id') else "general"
    
    # Process the symptoms with user ID for history tracking
    response_message = bot.process_user_symptoms(user_text, user_id)
    
    # Send response back to user
    bot.line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response_message)
    )

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'message': 'Qwen Hospital Recommendation Bot is running'}

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint for debugging"""
    if request.method == 'POST':
        data = request.get_json()
        user_text = data.get('text', 'ปวดหัว มีไข้')
        user_id = data.get('user_id', 'test_user')
        result = bot.process_user_symptoms(user_text, user_id)
        return {'input': user_text, 'user_id': user_id, 'output': result}
    else:
        return {'status': 'Test endpoint ready', 'usage': 'POST with {"text": "your symptoms", "user_id": "optional_user_id"}'}

@app.route('/history', methods=['GET'])
def get_history():
    """Get recommendation history (for debugging)"""
    return {'recommendation_history': bot.recommendation_history}

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """Reset recommendation history (for testing)"""
    data = request.get_json()
    user_id = data.get('user_id', 'all')
    
    if user_id == 'all':
        bot.recommendation_history = {}
        message = "All recommendation history cleared"
    else:
        if user_id in bot.recommendation_history:
            del bot.recommendation_history[user_id]
            message = f"History cleared for user {user_id}"
        else:
            message = f"No history found for user {user_id}"
    
    bot.save_recommendation_history()
    return {'message': message}

if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)