from flask import Flask, render_template, request, jsonify
from hug4 import TextSummarizer
import url2txt_final
import warnings
import sys
import os

app = Flask(__name__)

# Suppress all warnings and logs
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    youtube_url = data.get('url')

    if not youtube_url:
        return jsonify({"error": "No URL provided"}), 400

    transcript = url2txt_final.transcribe_youtube_video(youtube_url)
    
    if not transcript:
        return jsonify({"error": "Failed to transcribe video"}), 500

    summarizer = TextSummarizer()
    summary = summarizer.generate_summary(transcript)

    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
