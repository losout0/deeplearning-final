import re
from pathlib import Path
import urllib.request


GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/17005/pg17005.txt",
    "https://www.gutenberg.org/cache/epub/38496/pg38496.txt",
    "https://www.gutenberg.org/cache/epub/29040/pg29040.txt",
    "https://www.gutenberg.org/cache/epub/31552/pg31552.txt",
    "https://www.gutenberg.org/cache/epub/62383/pg62383.txt",
    "https://www.gutenberg.org/cache/epub/71031/pg71031.txt",
    "https://www.gutenberg.org/cache/epub/26777/pg26777.txt",
    "https://www.gutenberg.org/cache/epub/3333/pg3333.txt",
    "https://www.gutenberg.org/cache/epub/31509/pg31509.txt",
    "https://www.gutenberg.org/cache/epub/34387/pg34387.txt",
    "https://www.gutenberg.org/cache/epub/40409/pg40409.txt",
    "https://www.gutenberg.org/cache/epub/70819/pg70819.txt",
    "https://www.gutenberg.org/cache/epub/25840/pg25840.txt",
    "https://www.gutenberg.org/cache/epub/67767/pg67767.txt",
    "https://www.gutenberg.org/cache/epub/29484/pg29484.txt",
    "https://www.gutenberg.org/cache/epub/28691/pg28691.txt",
    "https://www.gutenberg.org/cache/epub/58689/pg58689.txt",
    "https://www.gutenberg.org/cache/epub/70764/pg70764.txt",
    "https://www.gutenberg.org/cache/epub/18220/pg18220.txt",
    "https://www.gutenberg.org/cache/epub/15047/pg15047.txt",
    "https://www.gutenberg.org/cache/epub/29435/pg29435.txt",
    "https://www.gutenberg.org/cache/epub/62624/pg62624.txt",
    "https://www.gutenberg.org/cache/epub/68932/pg68932.txt"
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
    DATA_DIR = Path("./data/ptbr"); DATA_DIR.mkdir(exist_ok=True)
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