def read_pdf(file):
    """
    Reads a PDF file and returns its content as a string.
    
    Note: This requires PyPDF2 to be added to requirements.txt
    """
    try:
        import io
        from PyPDF2 import PdfReader
        
        pdf_file = io.BytesIO(file.read())
        pdf_reader = PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except ImportError:
        raise ImportError("PyPDF2 is required to read PDF files. Add it to requirements.txt.")
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def read_docx(file):
    """
    Reads a DOCX file and returns its content as a string.
    
    Note: This requires python-docx to be added to requirements.txt
    """
    try:
        import io
        import docx
        
        docx_file = io.BytesIO(file.read())
        doc = docx.Document(docx_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    except ImportError:
        raise ImportError("python-docx is required to read DOCX files. Add it to requirements.txt.")
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {str(e)}")
