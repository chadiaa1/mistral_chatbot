# app.py - Version avec gestion optimisée des timeouts
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import json
from datetime import datetime
import logging
import os
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class CBTChatbot:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.max_history_length = 5  # Réduit pour des prompts plus courts
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.model_name = os.getenv('MODEL_NAME', 'mistral_cbt')
        
        # Réponses de fallback pour les cas de timeout
        self.fallback_responses = [
            "Je comprends que tu traverses une période difficile. Peux-tu me parler de ce que tu ressens en ce moment ?",
            "C'est important d'explorer ces sentiments. Qu'est-ce qui se passe dans ta tête quand tu vis cette situation ?",
            "Je t'entends. Prenons un moment pour identifier les pensées qui accompagnent ces émotions.",
            "Ces émotions sont valides. Comment pourrions-nous examiner les pensées qui les alimentent ?",
            "Merci de partager cela avec moi. Qu'est-ce qui te vient à l'esprit quand tu penses à cette situation ?"
        ]
        
        # Techniques CBT simplifiées pour réponses rapides
        self.quick_cbt_techniques = {
            "stress": "Quand tu te sens stressé, quelles pensées traversent ton esprit ? 🤔",
            "anxiété": "L'anxiété peut amplifier nos craintes. Quelles preuves as-tu que tes peurs vont se réaliser ?",
            "tristesse": "La tristesse est une émotion importante. Qu'est-ce qu'elle essaie de te dire ?",
            "colère": "La colère peut masquer d'autres émotions. Qu'y a-t-il sous cette colère ?",
            "culpabilité": "Tu sembles te blâmer. Et si on examinait tous les facteurs en jeu ?",
            "perfectionnisme": "Le perfectionnisme peut être épuisant. Qu'est-ce qui serait 'assez bien' ?",
            "catastrophe": "Tu imagines le pire scénario. Quelles autres issues sont possibles ?",
            "noir_blanc": "Tu vois les choses en tout ou rien. Où pourrait se situer le juste milieu ?"
        }
    
    def check_ollama_status(self) -> bool:
        """Vérifie si Ollama est accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def detect_keywords(self, message: str) -> str:
        """Détection simple et rapide de mots-clés émotionnels"""
        message_lower = message.lower()
        
        keyword_map = {
            "stress": ["stressé", "stress", "pression", "tendu"],
            "anxiété": ["anxieux", "angoisse", "inquiet", "nerveux"],
            "tristesse": ["triste", "déprimé", "mélancolique", "abattu"],
            "colère": ["colère", "énervé", "irrité", "furieux"],
            "culpabilité": ["coupable", "faute", "responsable", "blâme"],
            "perfectionnisme": ["parfait", "parfaitement", "impeccable"],
            "catastrophe": ["catastrophe", "terrible", "horrible", "affreux"],
            "noir_blanc": ["toujours", "jamais", "tout", "rien"]
        }
        
        for emotion, keywords in keyword_map.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion
        
        return "general"
    
    def generate_quick_response(self, user_message: str, detected_emotion: str) -> str:
        """Génère une réponse rapide basée sur les mots-clés détectés"""
        
        if detected_emotion in self.quick_cbt_techniques:
            base_response = self.quick_cbt_techniques[detected_emotion]
        else:
            # Utiliser une réponse de fallback aléatoire basée sur la longueur du message
            import hashlib
            hash_obj = hashlib.md5(user_message.encode())
            index = int(hash_obj.hexdigest(), 16) % len(self.fallback_responses)
            base_response = self.fallback_responses[index]
        
        # Ajouter une question de suivi contextuelle
        follow_up_questions = [
            "Peux-tu me décrire cette situation plus précisément ?",
            "Qu'est-ce qui rend cette situation particulièrement difficile pour toi ?",
            "Comment ton corps réagit-il dans ces moments ?",
            "Quelles pensées reviennent le plus souvent ?",
            "Y a-t-il un élément déclencheur que tu as remarqué ?"
        ]
        
        # Choisir une question de suivi basée sur l'émotion
        question_index = hash(detected_emotion) % len(follow_up_questions)
        follow_up = follow_up_questions[question_index]
        
        return f"{base_response}\n\n{follow_up}"
    
    def call_ollama_with_timeout(self, prompt: str, timeout: int = 15) -> Optional[str]:
        """Appel à Ollama avec timeout strict et fallback"""
        
        def make_request():
            try:
                response = requests.post(
                    f'{self.ollama_url}/api/generate',
                    json={
                        'model': self.model_name,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': 0.7,
                            'top_p': 0.9,
                            'num_predict': 150,  # Réduit pour des réponses plus rapides
                            'stop': ['[INST]', '</s>', '\n\n\n']
                        }
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()['response'].strip()
                    # Nettoyer la réponse
                    if result.startswith('</s>'):
                        result = result[4:].strip()
                    return result
                else:
                    logger.error(f"Erreur Ollama HTTP {response.status_code}")
                    return None
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'appel Ollama: {e}")
                return None
        
        # Utiliser ThreadPoolExecutor pour un timeout strict
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(make_request)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                logger.warning(f"Timeout Ollama après {timeout}s")
                return None
            except Exception as e:
                logger.error(f"Erreur inattendue: {e}")
                return None
    
    def generate_response(self, user_message: str) -> str:
        """Génère une réponse avec stratégie de fallback intelligente"""
        
        # Détection rapide d'émotion/thème
        detected_emotion = self.detect_keywords(user_message)
        
        # Vérifier l'état d'Ollama avant d'essayer
        if not self.check_ollama_status():
            logger.warning("Ollama n'est pas accessible, utilisation du fallback")
            response = self.generate_quick_response(user_message, detected_emotion)
        else:
            # Construire un prompt optimisé et court
            context = ""
            if self.conversation_history:
                # Prendre seulement le dernier échange pour réduire la taille
                last_exchange = self.conversation_history[-1]
                context = f"Contexte: L'utilisateur a dit '{last_exchange['user']}' et tu as répondu '{last_exchange['bot'][:100]}...'\n"
            
            optimized_prompt = f"""<s>[INST] Tu es un thérapeute CBT. {context}
L'utilisateur dit: "{user_message}"

Réponds brièvement (2-3 phrases max) en français avec empathie et une question thérapeutique. [/INST]"""
            
            # Essayer d'appeler Ollama avec timeout court
            start_time = time.time()
            ollama_response = self.call_ollama_with_timeout(optimized_prompt, timeout=15)
            call_duration = time.time() - start_time
            
            if ollama_response:
                logger.info(f"Réponse Ollama obtenue en {call_duration:.2f}s")
                response = ollama_response
                
                # Ajouter une technique CBT si la réponse est courte
                if len(response) < 100 and detected_emotion in self.quick_cbt_techniques:
                    response += f"\n\n💡 {self.quick_cbt_techniques[detected_emotion]}"
            else:
                logger.warning(f"Fallback après échec Ollama (durée: {call_duration:.2f}s)")
                response = self.generate_quick_response(user_message, detected_emotion)
        
        # Sauvegarder dans l'historique
        self.conversation_history.append({
            'user': user_message,
            'bot': response,
            'timestamp': datetime.now().isoformat(),
            'detected_emotion': detected_emotion,
            'source': 'ollama' if 'ollama_response' in locals() and ollama_response else 'fallback'
        })
        
        # Limiter l'historique
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return response

# Instance globale
chatbot = CBTChatbot()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message vide'}), 400
        
        if len(user_message) > 500:  # Limite plus stricte
            return jsonify({'error': 'Message trop long (500 caractères max)'}), 400
        
        # Mesurer le temps de réponse
        start_time = time.time()
        response = chatbot.generate_response(user_message)
        response_time = time.time() - start_time
        
        logger.info(f"Réponse générée en {response_time:.2f}s")
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'response_time': round(response_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /chat: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    try:
        chatbot.conversation_history = []
        return jsonify({'message': 'Conversation réinitialisée'})
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation: {e}")
        return jsonify({'error': 'Erreur lors de la réinitialisation'}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint pour vérifier l'état du système"""
    ollama_status = chatbot.check_ollama_status()
    return jsonify({
        'ollama_connected': ollama_status,
        'model': chatbot.model_name,
        'conversation_length': len(chatbot.conversation_history),
        'last_activity': chatbot.conversation_history[-1]['timestamp'] if chatbot.conversation_history else None
    })

if __name__ == '__main__':
    # Vérifier Ollama au démarrage
    if chatbot.check_ollama_status():
        logger.info("✅ Ollama est accessible")
    else:
        logger.warning("⚠️  Ollama n'est pas accessible - mode fallback activé")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)