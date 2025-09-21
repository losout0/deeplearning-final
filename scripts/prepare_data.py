import os
import re
import glob
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
CONSOLIDATED_TEXT_PATH = 'data/processed/machado_consolidado.txt'
MIN_PARAGRAPH_LENGTH = 50 

def clean_gutenberg_text(text):
    start_marker = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*"
    end_marker = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*"
    
    start_match = re.search(start_marker, text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
        
    end_match = re.search(end_marker, text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
        
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    text = text + "<|endoftext|>"
    
    return text

def save_split(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(data))
    print(f"Salvo {len(data)} parágrafos em: {path}")

def main():
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    all_cleaned_texts = []
    file_paths = glob.glob(os.path.join(RAW_DATA_DIR, '*.txt'))
    file_paths.sort()
    
    if not file_paths:
        print(f"ERRO: Nenhum arquivo .txt encontrado em '{RAW_DATA_DIR}'.")
        print("Por favor, execute 'python scripts/download_data.py' primeiro.")
        return

    print(f"Encontrados {len(file_paths)} arquivos em '{RAW_DATA_DIR}'.")
    
    for filepath in file_paths:
        print(f"  -> Processando: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
                cleaned_content = clean_gutenberg_text(raw_content)
                if cleaned_content:
                    all_cleaned_texts.append(cleaned_content)
        except Exception as e:
            print(f"    AVISO: Falha ao ler ou processar o arquivo {filepath}. Erro: {e}")

    consolidated_text = "\n\n".join(all_cleaned_texts)
    
    with open(CONSOLIDATED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(consolidated_text)
    print(f"\nCorpus consolidado com ~{len(consolidated_text.split())} palavras salvo em: {CONSOLIDATED_TEXT_PATH}")

    print("\n--- Dividindo o corpus em treino, validação e teste ---")

    paragraphs = [p.strip() for p in consolidated_text.split('\n\n') if len(p.strip()) >= MIN_PARAGRAPH_LENGTH]
    
    if len(paragraphs) < 10:
        print("ERRO: O corpus consolidado tem muito poucos parágrafos para ser dividido.")
        return

    train_val, test = train_test_split(paragraphs, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=(1/9), random_state=42)

    save_split(train, os.path.join(PROCESSED_DATA_DIR, "train.txt"))
    save_split(val, os.path.join(PROCESSED_DATA_DIR, "val.txt"))
    save_split(test, os.path.join(PROCESSED_DATA_DIR, "test.txt"))
    
    print("\n--- Preparação de dados concluída com sucesso! ---")

if __name__ == "__main__":
    main()