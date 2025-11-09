import os
import zipfile
import requests
import shutil
import subprocess
import sys

def instalar_ffmpeg():
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    destino_zip = "ffmpeg.zip"
    pasta_destino = "C:\\ffmpeg"

    print("🔽 Baixando FFmpeg (pode levar alguns minutos)...")
    response = requests.get(ffmpeg_url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(destino_zip, "wb") as file:
        baixado = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                baixado += len(chunk)
                file.write(chunk)
                done = int(50 * baixado / total)
                sys.stdout.write(f"\r[{'█' * done}{'.' * (50 - done)}] {baixado//1024} KB")
                sys.stdout.flush()

    print("\n📦 Extraindo FFmpeg...")
    with zipfile.ZipFile(destino_zip, "r") as zip_ref:
        zip_ref.extractall("ffmpeg_temp")

    # A pasta extraída tem nome variável, então detectamos automaticamente
    nome_pasta = next(os.walk("ffmpeg_temp"))[1][0]
    origem = os.path.join("ffmpeg_temp", nome_pasta)
    
    # Move o conteúdo para C:\ffmpeg
    if os.path.exists(pasta_destino):
        shutil.rmtree(pasta_destino)
    shutil.move(origem, pasta_destino)

    print("🧩 Adicionando FFmpeg ao PATH do sistema...")
    # Adiciona permanentemente ao PATH do Windows
    subprocess.run(
        f'setx PATH "%PATH%;{pasta_destino}\\bin"',
        shell=True
    )

    # Limpa arquivos temporários
    os.remove(destino_zip)
    shutil.rmtree("ffmpeg_temp")

    print("\n✅ FFmpeg instalado com sucesso em:", pasta_destino)
    print("➡️ Reinicie o terminal e teste com: ffprobe -version")

if __name__ == "__main__":
    instalar_ffmpeg()
