import re
from pathlib import Path
import urllib.request


GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/17005/pg17005.txt",

]


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

    if len(text) > 491: text = text[491:]
    if len(text) > 18413: text = text[:-18413]

    text = re.sub(r"[ \t]+", " ", text)      # múltiplos espaços → espaço único
    text = re.sub(r"\n{3,}", "\n\n", text)   # múltiplas quebras → no máx. duas
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def download_corpus(urls, output_dir):
    saved_files = []

    for idx, url in enumerate(urls, 1):
        file_path = output_dir / f"book_{idx:02d}.txt"
        
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(file_path, "wb") as f:
                f.write(response.read())

            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_text=clean_text(raw_text)
            file_path.write_text(cleaned_text, encoding="utf-8")
            saved_files.append(str(file_path))
            print(f"[OK] {url} -> {file_path}")
            
        except Exception as error:
            print(f"[ERRO] {url} ({error})")

    concat = ""
    for path in saved_files:
        concat += Path(path).read_text(encoding="utf-8") + " <|endoftext|> "

    return concat


def get_data():
    DATA_DIR = Path("./data/ptbr"); DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = download_corpus(urls=GUTENBERG_URLS, output_dir=DATA_DIR)
    data = data[0: len(data) // 2]
    
    n = len(data)
    train_end = int(0.8 * n)
    test_end = int(0.9 * n)

    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    val_data = data[test_end:]

    with open("train.txt", "w", encoding="utf-8") as f:
        f.write(train_data)

    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(test_data)

    with open("val.txt", "w", encoding="utf-8") as f:
        f.write(val_data)

    print(f"Treino: {len(train_data)} docs")
    print(f"Teste: {len(test_data)} docs")
    print(f"Validação: {len(val_data)} docs")
    
    return train_data, test_data, val_data