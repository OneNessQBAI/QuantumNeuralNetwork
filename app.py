from flask import Flask, request, jsonify, render_template, send_from_directory
from main_workflow import process_user_input
from Qwen import Qwen
from Qwen2Math import Qwen2Math
from DeepSeekCoder import DeepSeekCoder
from Starcoder import Starcoder
from quantum_algorithms.quantum_optimization import QuantumOptimization
from synthetic_data_layers.data_layer_qwen_qwen2_math import DataLayerQwenQwen2Math
from synthetic_data_layers.data_layer_qwen2_math_tocoder import DataLayerQwen2MathToCoder
from synthetic_data_layers.data_layer_tocoder_starcoder import DataLayerToCoderStarcoder
from system_prompts import QWEN_PROMPT, QWEN2_MATH_PROMPT, TOCODER_PROMPT, STARCODER_PROMPT
from openai import OpenAI
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize models and data layers
qwen5b = None
qwen2_math = None
qwen2_5_coder = None
starcoder = None
deepseek = None

data_layer_qwen_qwen2_math = DataLayerQwenQwen2Math()
data_layer_qwen2_math_tocoder = DataLayerQwen2MathToCoder()
data_layer_tocoder_starcoder = DataLayerToCoderStarcoder()

quantum_optimizer = QuantumOptimization()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect_api', methods=['POST'])
def connect_api():
    global qwen5b, qwen2_math, qwen2_5_coder, starcoder, deepseek
    data = request.json
    api_key = data['api_key']
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the API key
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, World!"}],
            max_tokens=5
        )
        app.logger.info(f"API test response: {response}")
        
        # Initialize models with the API key and system prompts
        try:
            qwen5b = Qwen(system_prompt=QWEN_PROMPT, api_key=api_key)
            app.logger.info("Qwen5b initialized with QWEN_PROMPT")
            
            qwen2_math = Qwen2Math(system_prompt=QWEN2_MATH_PROMPT, api_key=api_key)
            app.logger.info("Qwen2Math initialized with QWEN2_MATH_PROMPT")
            
            qwen2_5_coder = DeepSeekCoder(system_prompt=TOCODER_PROMPT, api_key=api_key)
            app.logger.info("Qwen2.5Coder initialized with TOCODER_PROMPT")
            
            starcoder = Starcoder(system_prompt=STARCODER_PROMPT, api_key=api_key)
            app.logger.info("Starcoder initialized with STARCODER_PROMPT")
            
            deepseek = DeepSeekCoder(system_prompt=TOCODER_PROMPT, api_key=api_key)
            app.logger.info("DeepSeek initialized with TOCODER_PROMPT")
        except Exception as e:
            app.logger.error(f"Error initializing models: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to initialize models'}), 500
        
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error connecting API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/qnn_process', methods=['POST'])
def qnn_process():
    api_key = request.headers.get('X-API-KEY')
    if not api_key or not qwen5b:
        return jsonify({'error': 'API key not set or models not initialized'}), 401

    data = request.json
    user_input = data['message']

    try:
        response = process_user_input(
            user_input, qwen5b, qwen2_math, qwen2_5_coder,
            starcoder, deepseek,
            data_layer_qwen_qwen2_math,
            data_layer_qwen2_math_tocoder,
            data_layer_tocoder_starcoder,
            quantum_optimizer
        )
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

# New route to serve images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# New route to serve video
@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)