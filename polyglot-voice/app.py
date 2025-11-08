import gradio as gr
from deep_translator import GoogleTranslator
from transformers import pipeline
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import pygame
from gtts import gTTS
import os

# --- Carregar variáveis de ambiente ---
load_dotenv()

# Tentar importar OpenAI (versão nova ou antiga)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    USE_NEW_API = True
    print("OpenAI API v1.0+ carregada com sucesso.")
except ImportError:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    USE_NEW_API = False
    print("OpenAI API antiga carregada com sucesso.")

# --- Função 1: Grava áudio ---
def gravar_audio(duracao=5, fs=16000):
    print("Gravando... fale algo!")
    audio = sd.rec(int(duracao * fs), samplerate=fs, channels=1)
    sd.wait()
    write("entrada.wav", fs, audio)
    print("Gravação concluída.")
    return "entrada.wav"

# --- Função 2: Transcreve fala (Whisper) ---
def transcrever(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            if USE_NEW_API:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
            else:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript["text"]
    except Exception as e:
        return f"Erro na transcrição: {e}"

# --- Função 3: Tradução automática ---
def traduzir(texto):
    try:
        tradutor_en = GoogleTranslator(source='auto', target='en')
        traducao_en = tradutor_en.translate(texto)
        tradutor_pt = GoogleTranslator(source='auto', target='pt')
        traducao_pt = tradutor_pt.translate(texto)
        
        idioma_origem = "pt" if traducao_en != texto else "en"
        traducao_final = traducao_en if idioma_origem == "pt" else traducao_pt
        idioma_destino = "en" if idioma_origem == "pt" else "pt"
        
        return traducao_final, idioma_origem, idioma_destino
    except Exception as e:
        return f"Erro na tradução: {e}", "desconhecido", "desconhecido"

# --- Função 4: Análise de Sentimento ---
sentimento_model = pipeline("sentiment-analysis")

def analisar_sentimento(texto):
    try:
        resultado = sentimento_model(texto)[0]
        label = resultado["label"]
        score = resultado["score"]
        
        if "POS" in label:
            status = "Positivo"
        elif "NEG" in label:
            status = "Negativo"
        else:
            status = "Neutro"
        
        return f"Status: {status} | {label} ({score:.2f})"
    except Exception as e:
        return f"Erro na análise de sentimento: {e}"

# --- Função 5: Fala (TTS) ---
def falar_texto(texto, idioma_destino):
    try:
        idioma_tts = "pt" if idioma_destino == "pt" else "en"
        tts = gTTS(text=texto, lang=idioma_tts)
        tts.save("saida.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("saida.mp3")
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        return "saida.mp3"
    except Exception as e:
        print(f"Erro no TTS: {e}")
        return None

# --- Função 6: Pipeline completo ---
def processar_audio(duracao):
    try:
        audio = gravar_audio(duracao)
        texto = transcrever(audio)
        
        if "Erro" in texto:
            return texto, None
        
        traducao, idioma_origem, idioma_destino = traduzir(texto)
        sentimento = analisar_sentimento(texto)
        arquivo_audio = falar_texto(traducao, idioma_destino)
        
        saida = (
            f"### Texto Original ({idioma_origem}):\n"
            f"{texto}\n\n"
            f"### Tradução ({idioma_destino}):\n{traducao}\n\n"
            f"### Sentimento:\n{sentimento}"
        )
        
        return saida, arquivo_audio
    except Exception as e:
        return f"Erro no processamento: {e}", None

# --- Interface Gradio com gr.Interface (evita o bug) ---
interface = gr.Interface(
    fn=processar_audio,
    inputs=gr.Slider(3, 10, value=5, step=1, label="Duração da Gravação (segundos)"),
    outputs=[
        gr.Markdown(label="Resultado"),
        gr.Audio(label="Tradução Falada", type="filepath")
    ],
    title="🎙️ Polyglot Voice — Tradutor, Sentimento e Voz",
    description="Grave sua voz, traduza automaticamente e ouça o resultado!",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(share=True, enable_queue=False)