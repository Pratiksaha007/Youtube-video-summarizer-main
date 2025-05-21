from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import re
from typing import List, Tuple, Dict
import url2txt_final
import warnings

warnings.filterwarnings("ignore")

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Download required NLTK data
        for resource in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        
        self.stopwords = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and structure the input text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters while keeping essential punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Split into sentences and rejoin with proper spacing
        sentences = sent_tokenize(text)
        
        # Remove redundant sentences
        unique_sentences = []
        seen_sentences = set()
        for sentence in sentences:
            # Create a normalized version for comparison
            normalized = ' '.join(word_tokenize(sentence.lower()))
            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
        
        return ' '.join(unique_sentences)
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract important phrases from text."""
        words = word_tokenize(text.lower())
        # Remove stopwords and short words
        keywords = [word for word in words if word not in self.stopwords and len(word) > 3]
        # Get frequency distribution
        freq_dist = nltk.FreqDist(keywords)
        # Return top phrases
        return [word for word, _ in freq_dist.most_common(10)]
    
    def calculate_optimal_lengths(self, text: str) -> Tuple[int, int]:
        """Calculate optimal summary length based on input text."""
        word_count = len(text.split())
        
        # Dynamic scaling based on input length
        if word_count <= 100:
            ratio = 0.6  # Preserve more content for short texts
        elif word_count <= 500:
            ratio = 0.4
        else:
            ratio = 0.3  # More aggressive summarization for longer texts
            
        max_length = int(word_count * ratio)
        max_length = max(150, min(512, max_length))  # Enforce bounds
        min_length = max(50, int(max_length * 0.5))
        
        return min_length, max_length
    
    def optimize_generation_params(self, text: str) -> Dict:
        """Optimize generation parameters based on text characteristics."""
        word_count = len(text.split())
        
        params = {
            'num_beams': 4,
            'temperature': 0.7,
            'repetition_penalty': 1.2,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
        }
        
        # Adjust parameters based on text length
        if word_count > 500:
            params['num_beams'] = 5
            params['temperature'] = 0.6  # More focused generation for longer texts
            params['repetition_penalty'] = 1.3
        
        return params
    
    def ensure_summary_coherence(self, summary: str) -> str:
        """Ensure the summary is coherent and well-formed."""
        # Fix incomplete sentences
        if not summary.endswith(('.', '!', '?')):
            summary = summary.rstrip() + '.'
        
        # Ensure proper capitalization
        sentences = sent_tokenize(summary)
        sentences = [s.capitalize() for s in sentences]
        
        return ' '.join(sentences)
    
    def generate_summary(self, text: str) -> str:
        """Generate an improved summary of the input text."""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract key phrases to guide generation
        key_phrases = self.extract_key_phrases(processed_text)
        
        # Calculate optimal lengths
        min_length, max_length = self.calculate_optimal_lengths(processed_text)
        
        # Get optimized generation parameters
        gen_params = self.optimize_generation_params(processed_text)
        
        # Prepare input text with key phrases emphasis
        inputs = self.tokenizer(processed_text, return_tensors="pt",max_length=1024, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate summary with optimized parameters
        summary_ids = self.model.generate(
            inputs['input_ids'],
            min_length=min_length,
            max_length=max_length,
            **gen_params,
            early_stopping=True,
            do_sample=True 
        )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process summary
        summary = self.ensure_summary_coherence(summary)
        
        return summary

def main(youtube_url):
    # Example usage
    text = url2txt_final.transcribe_youtube_video(youtube_url)
    print("Summarizing...")
    
    summarizer = TextSummarizer()
    summary = summarizer.generate_summary(text)
    
    print("Original Length:", len(text.split()))
    print("Summary Length:", len(summary.split()))
    print("\nSummary:", summary)

if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ")
    main(youtube_url)