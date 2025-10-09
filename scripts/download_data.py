import os
import requests
from tqdm import tqdm

BOOKS = {
    "dom_casmurro.txt": "https://www.gutenberg.org/files/55752/55752-0.txt",
    "memorias_postumas_de_bras_cubas.txt": "https://www.gutenberg.org/files/54829/54829-0.txt",
    "poesias_completas.txt": "https://www.gutenberg.org/files/61653/61653-0.txt",
    "quincas_borba.txt": "https://www.gutenberg.org/files/55682/55682-0.txt",
    "esau_e_jaco.txt": "https://www.gutenberg.org/cache/epub/56737/pg56737.txt",
    "papeis_avulsos.txt": "https://www.gutenberg.org/cache/epub/57001/pg57001.txt",
    "helena.txt": "https://www.gutenberg.org/cache/epub/67162/pg67162.txt",
    "historias_sem_data.txt": "https://www.gutenberg.org/cache/epub/33056/pg33056.txt",
    "a_mao_e_a_luva.txt": "https://www.gutenberg.org/cache/epub/53101/pg53101.txt",
    "reliquias_de_casa_velha.txt": "https://www.gutenberg.org/files/67935/67935-0.txt",
    "memorial_de_ayres.txt": "https://www.gutenberg.org/files/55797/55797-0.txt",
    "iaia_garcia.txt": "https://www.gutenberg.org/files/67780/67780-0.txt",
}

OUTPUT_DIR = "data/raw"

def download_files():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for filename, url in BOOKS.items():
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            progress_bar = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=f"Baixando {filename}"
            )
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print(f"  -> AVISO: O tamanho do download de {filename} não corresponde ao esperado.")

        except requests.exceptions.RequestException as e:
            print(f"  -> ERRO ao baixar {filename}: {e}")

    print("\nDownload de todos os arquivos concluído!")

if __name__ == "__main__":
    download_files()