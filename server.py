from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from transformers import pipeline
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from joblib import dump, load


app = Flask(__name__)
CORS(app)  # Apply CORS to all routes

#----------------------------------------------------------------------
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


def generate_summary(input_text):
    # Preprocess the input text
    preprocessed_input = preprocess_text(input_text)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    model = load('summaryModel.joblib')
    # Tokenization
    input_tokenized = tokenizer(preprocessed_input, return_tensors="pt")

    # Model inference
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(input_tokenized['input_ids'], max_length=50, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary



#-------------------------------------------------------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file.filename.endswith('.pdf'):
        
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages) 
        text=""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text =generate_summary(text)
        return jsonify({'text': text})
    else:
        return jsonify({'error': 'Unsupported file format'})

if __name__ == '__main__':
    app.run()
