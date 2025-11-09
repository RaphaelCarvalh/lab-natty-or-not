import gradio as gr
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from deep_translator import GoogleTranslator
from transformers import pipeline
from gtts import gTTS
import pygame
import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import shutil

gravando = False
gravacao = None

load_dotenv()

sentimento_model = pipeline("sentiment-analysis")

def analisar_sentimento(texto):
    try:
        resultado = sentimento_model(texto)[0]
        label = resultado["label"].upper()
        score = resultado["score"]

        if "POS" in label:
            status = "positivo"
        elif "NEG" in label:
            status = "negativo"
        else:
            status = "neutro"

        return {"status": status, "score": score}
    except Exception as e:
        return {"status": "erro", "score": 0.0}

def listar_microfones():
    dispositivos = sd.query_devices()
    mics = [f"{i}: {d['name']}" for i, d in enumerate(dispositivos) if d['max_input_channels'] > 0]
    return mics

# --- Função de gravação controlada ---
def gravar_audio_manual(action, mic_label, fs=16000):
    global gravando, gravacao

    if mic_label == "Nenhum microfone encontrado":
        return "Nenhum microfone disponível.", None

    # Extrai índice do microfone selecionado (ex: "2: Microfone XYZ")
    mic_index = int(mic_label.split(":")[0])

    if action == "start":
        gravando = True
        gravacao = sd.rec(int(60 * fs), samplerate=fs, channels=1, device=mic_index, dtype='float32')
        return f"Gravando com {mic_label}. Clique em Parar para encerrar.", None

    elif action == "stop":
        if not gravando:
            return "Nenhuma gravação ativa.", None

        sd.stop()
        gravando = False

        # Normaliza o volume
        if gravacao is not None and np.max(np.abs(gravacao)) > 0:
            audio_norm = gravacao / np.max(np.abs(gravacao))
            audio_norm = (audio_norm * 32767).astype(np.int16)
            write("entrada.wav", fs, audio_norm)
            return "Gravação concluída e salva como entrada.wav", "entrada.wav"
        else:
            return "Erro: Nenhum áudio captado.", None

# --- Transcrição com Whisper local ---
def transcrever_audio(arquivo):
    try:
        if arquivo is None:
            return "Nenhum áudio fornecido.", None

        # Extrai o caminho real dependendo do tipo do input
        if isinstance(arquivo, dict) and "name" in arquivo:
            caminho_original = arquivo["name"]
        elif isinstance(arquivo, str):
            caminho_original = arquivo
        else:
            return "Formato de arquivo inválido.", None

        if not os.path.exists(caminho_original):
            return "Arquivo de áudio não encontrado.", None

        # Copia para um caminho seguro antes de o arquivo temporário ser limpo
        caminho_local = "entrada_temp.wav"
        shutil.copy(caminho_original, caminho_local)

        print(f"Transcrevendo arquivo: {caminho_local}")
        model = whisper.load_model("base")
        result = model.transcribe(caminho_local, language="pt")

        texto = result.get("text", "").strip()
        if not texto:
            return "Nenhum texto detectado (áudio silencioso ou muito curto).", None

        print("Transcrição concluída com sucesso!")
        return f"Transcrição: {texto}", texto

    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return f"Erro na transcrição: {e}", None

# --- Tradução ---
def traduzir_texto(texto):
    try:
        if not texto or texto.strip() == "":
            return "Nenhum texto para traduzir.", None

        # Detecta automaticamente o idioma e traduz para o oposto (PT <-> EN)
        tradutor = GoogleTranslator(source='auto', target='en')
        traducao = tradutor.translate(texto)

        # Verifica se o texto já está em inglês
        if traducao.strip().lower() == texto.strip().lower():
            tradutor = GoogleTranslator(source='auto', target='pt')
            traducao = tradutor.translate(texto)
            idioma_alvo = "pt"
        else:
            idioma_alvo = "en"

        return f"Tradução ({idioma_alvo.upper()}): {traducao}", traducao

    except Exception as e:
        return f"Erro na tradução: {e}", None

# --- TTS ---
def falar_texto(texto):
    try:
        if not texto or texto.strip() == "":
            return "Nenhum texto para converter em fala.", None

        # 1️⃣ Detecta sentimento
        sentimento = analisar_sentimento(texto)
        status = sentimento["status"]
        score = sentimento["score"]

        # 2️⃣ Ajusta a “emoção” da fala
        if status == "positivo":
            lang = "en"
            mensagem = f"(Tom alegre, confiança {score:.2f})"
            velocidade = 1.2  # fala mais rápida
        elif status == "negativo":
            lang = "en"
            mensagem = f"(Tom triste, confiança {score:.2f})"
            velocidade = 0.8  # fala mais lenta
        else:
            lang = "en"
            mensagem = f"(Tom neutro, confiança {score:.2f})"
            velocidade = 1.0  # normal

        # 3️⃣ Cria o arquivo de áudio
        tts = gTTS(text=texto, lang=lang)
        tts.save("saida.mp3")

        # 4️⃣ Reproduz com pygame ajustando velocidade
        pygame.mixer.init()
        pygame.mixer.music.load("saida.mp3")
        pygame.mixer.music.play()

        # 5️⃣ Simula modulação pela duração (para simular emoção)
        duracao = os.path.getsize("saida.mp3") / 16000
        ajuste = max(0.5, min(1.5, velocidade))
        pygame.time.delay(int(duracao * 1000 / ajuste))

        return f"Áudio gerado ({mensagem})", "saida.mp3"

    except Exception as e:
        return f"Erro no TTS: {e}", None

def play_audio_traducao(arquivo):
    """Toca o áudio gerado pela tradução"""
    try:
        if not arquivo or not os.path.exists(arquivo):
            return "Nenhum áudio disponível para reprodução."
        pygame.mixer.init()
        pygame.mixer.music.load(arquivo)
        pygame.mixer.music.play()
        return f"Tocando: {os.path.basename(arquivo)}"
    except Exception as e:
        return f"Erro ao tocar áudio: {e}"

def stop_audio_traducao():
    """Interrompe o áudio que estiver tocando"""
    try:
        pygame.mixer.music.stop()
        return "Reprodução parada."
    except Exception as e:
        return f"Erro ao parar áudio: {e}"

# --- Interface com controle manual ---
with gr.Blocks(title="Polyglot Voice") as demo:
    gr.Markdown("## Polyglot Voice — Tradutor, Transcrição e Voz")

    # Seleção de microfone
    mic_opcoes = listar_microfones()
    if not mic_opcoes:
        mic_opcoes = ["Nenhum microfone encontrado"]

    mic_selector = gr.Dropdown(choices=mic_opcoes, value=mic_opcoes[0], label="Escolha o microfone")

    # Linha: status e áudio gravado
    with gr.Row():
        status = gr.Textbox(label="Status da Gravação", interactive=False)
        audio_output = gr.Audio(label="Áudio Gravado", type="filepath")

    # Estados internos
    acao_start = gr.State("start")
    acao_stop = gr.State("stop")

    # Botões de controle de gravação
    with gr.Row():
        btn_start = gr.Button("Iniciar Gravação")
        btn_stop = gr.Button("Parar Gravação")

    btn_start.click(
        fn=gravar_audio_manual,
        inputs=[acao_start, mic_selector],
        outputs=[status, audio_output]
    )

    btn_stop.click(
        fn=gravar_audio_manual,
        inputs=[acao_stop, mic_selector],
        outputs=[status, audio_output]
    )

    # Transcrição e tradução
    with gr.Row():
        btn_transcrever = gr.Button("Transcrever e Traduzir")
        texto_saida = gr.Textbox(label="Texto / Tradução", interactive=False)
        audio_saida = gr.Audio(label="Tradução Falada", type="filepath")

    btn_transcrever.click(
        fn=transcrever_audio,
        inputs=audio_output,
        outputs=[texto_saida, texto_saida]
    )

    btn_transcrever.click(
        fn=traduzir_texto,
        inputs=texto_saida,
        outputs=[texto_saida, texto_saida]
    )

    btn_transcrever.click(
        fn=analisar_sentimento,
        inputs=texto_saida,
        outputs=[]  # não precisa exibir o resultado, apenas influenciar o TTS
    )

    btn_transcrever.click(
        fn=falar_texto,
        inputs=texto_saida,
        outputs=[texto_saida, audio_saida]
    )

    # Controle de reprodução do áudio gerado
    with gr.Row():
        btn_play = gr.Button("Reproduzir Tradução")
        btn_stop_audio = gr.Button("Parar Reprodução")
        status_audio = gr.Textbox(label="Status da Reprodução", interactive=False)

    btn_play.click(
        fn=play_audio_traducao,
        inputs=audio_saida,
        outputs=status_audio
    )

    btn_stop_audio.click(
        fn=stop_audio_traducao,
        outputs=status_audio
    )

demo.launch(share=True)

