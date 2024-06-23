import PyPDF2
import glob


def load_pdf_files_from_directory(directory):
    text = ""
    for file in glob.glob(directory + "/*.pdf"):
        if file.endswith('.pdf'):
            file_reader = PyPDF2.PdfReader(file)
            for page in file_reader.pages:
                text += page.extract_text()
    return text
