import fitz  # PyMuPDF
from transformers import pipeline
import re

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to clean extracted text
def clean_text(text):
    # Remove multiple newlines and extra spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove trailing spaces
    text = text.strip()
    return text

# Function to split text into chunks
def split_text(text, chunk_size=1024):
    sentences = text.split('. ')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) + 1 <= chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Function to format the summary
def format_summary(summary):
    # Capitalize the first letter of each sentence
    summary = '. '.join(sentence.capitalize() for sentence in summary.split('. '))
    # Add paragraph breaks
    paragraphs = summary.split('\n')
    formatted_summary = '\n\n'.join(paragraph.strip() for paragraph in paragraphs)
    return formatted_summary

# Function to summarize text using a transformer model
def summarize_text(text, model_name="facebook/bart-large-cnn", max_length=512, min_length=50):
    summarizer = pipeline("summarization", model=model_name)
    chunks = split_text(text)
    # Limit the number of chunks to summarize
    #chunks = chunks[:3]
    summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

# Main function to extract text from PDF and summarize it
def summarize_pdf(pdf_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Clean the extracted text
    clean_text_content = clean_text(text)
    
    # Summarize the cleaned text
    summary = summarize_text(clean_text_content)
    
    # Format the summary for better readability
    formatted_summary = format_summary(summary)
    
    return formatted_summary

# Path to the PDF file
pdf_path = 'your_pdf_file.pdf'

# Summarize the PDF and print the summary
summary = summarize_pdf(pdf_path)
print("Summary:")
print(summary)
