import json
import threading
import websocket
import pydub
import os
import base64

# Replace with your AssemblyAI API key
YOUR_API_TOKEN = os.environ.get("ASSEMBLYAI_API_KEY")
FRAMES_PER_BUFFER = 3200
CHANNELS = 1
SAMPLE_RATE = 16000

# Global variables to store the transcripts and timestamps
all_transcripts = []
all_timestamps = []

# Function to convert the MP4 to WAV PCM16
def convert_to_wav(filename):
    audio = pydub.AudioSegment.from_file(filename)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
    audio.export("converted.wav", format="wav")
    return "converted.wav"

# Callback functions for the WebSocket
def on_message(ws, message):
    global all_transcripts, all_timestamps
    transcript = json.loads(message)

    if transcript["message_type"] == 'FinalTranscript':
        text = transcript['text']
        start_time = transcript['audio_start']
        end_time = transcript['audio_end']
        all_transcripts.append(text)
        all_timestamps.append({'start': start_time, 'end': end_time})
        print(f"Final transcript received: {text}")

def on_open(ws):
    def send_data():
        with open("converted.wav", "rb") as f:
            while True:
                data = f.read(FRAMES_PER_BUFFER)
                if not data:
                    break
                data = base64.b64encode(data).decode("utf-8")
                ws.send(json.dumps({"audio_data": str(data)}))
        ws.send(json.dumps({"terminate_session": True}))
    
    threading.Thread(target=send_data).start()

def on_error(ws, error):
    print(error)
    save_transcripts()

def on_close(ws):
    print("WebSocket closed")
    save_transcripts()

def save_transcripts():
    transcript_data = {
        "text": " ".join(all_transcripts),
        "sentences": [
            {"start": timestamp['start'], "end": timestamp['end'], "sentence_text": all_transcripts[i]}
            for i, timestamp in enumerate(all_timestamps)
        ]
    }
    with open("transcripts.json", "w") as file:
        json.dump(transcript_data, file)

# Convert the MP4 file to WAV
input_filename = "path_to_file"
converted_filename = convert_to_wav(input_filename)

# Set up and run the WebSocket
websocket.enableTrace(False)
ws = websocket.WebSocketApp(
    f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={SAMPLE_RATE}",
    header={"Authorization": YOUR_API_TOKEN},
    on_message=on_message,
    on_open=on_open,
    on_error=on_error,
    on_close=on_close
)
ws.run_forever()
