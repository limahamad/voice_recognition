import speech_recognition as sr
import pyttsx3
import numpy as np
import keyboard
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from demo_utils import *
from pathlib import Path
import librosa
import csv
import soundmeter
import time
from faker import Faker
import mysql.connector
from dotenv import load_dotenv
import pyaudio
import sounddevice as sd

# load_dotenv()
# myDB = mysql.connector.connect(host = "localhost", user = "root", passwd = "Bibigullima#3")
# myCur = myDB.cursor()
# myCur.execute("use new_database")
# create_table_query = """
#         CREATE TABLE embeds (
#             name VARCHAR(255)  PRIMARY KEY
#         );
#         """
# # myCur.execute(create_table_query)
# print('Table created successfully')

# insert_q = """
#     insert into embeds values(0, 'lima');
#     """
# # myCur.execute(insert_q)
# myCur.execute("SELECT * FROM embeds")
# rows = myCur.fetchall()
# # Display the fetched rows
# for row in rows:
#     print(row)



# myDB.commit()
# myCur.close()
# myDB.close()

def get_active_output_device_index():
    p = pyaudio.PyAudio()
    active_index = p.get_default_output_device_info()['index']
    return active_index


def get_physical_output_devices():
    p = pyaudio.PyAudio()
    physical_devices = []
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxOutputChannels'] > 0:
            physical_devices.append((dev_info['index'], dev_info['name']))
    return physical_devices


def get_active_physical_output_device():
    active_index = get_active_output_device_index()
    physical_devices = get_physical_output_devices()
    for device_index, device_name in physical_devices:
        if device_index == active_index:
            return device_name
    return None


def get_active_input_device_index():
    p = pyaudio.PyAudio()
    active_index = p.get_default_input_device_info()['index']
    return active_index


def get_physical_input_devices():
    p = pyaudio.PyAudio()
    physical_devices = []
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            physical_devices.append((dev_info['index'], dev_info['name']))
    return physical_devices


def get_active_physical_input_device():
    active_index = get_active_input_device_index()
    physical_devices = get_physical_input_devices()
    for device_index, device_name in physical_devices:
        if device_index == active_index:
            return device_name
    return None


speaker_embeds = []
speaker_names = []
X = []
y = []
recognizer = sr.Recognizer()

speaker_names_file = "speaker_names.csv"
speaker_embeds_file = "speaker_embeds.csv"

output_file = "recorded_audio.mp3"
output_file2 = "recorded_audio2.mp3"


def generate_random_text(num_sentences=5):
    fake = Faker()
    random_text = ""
    for _ in range(num_sentences):
        random_text += fake.sentence() + " "
    return random_text


def store_stored_embeddings(speaker_names_file, speaker_embeds_file, speaker_embeds, speaker_names):
    with open(speaker_names_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for string in speaker_names:
            writer.writerow([string])

    with open(speaker_embeds_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # for num in speaker_embeds:
        #     writer.writerow([num])
        writer.writerows(speaker_embeds)


def load_stored_embeddings(speaker_names_file, speaker_embeds_file):
    string_array = []
    float_array = []
    with open(speaker_names_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            string_array.append(row[0])

    with open(speaker_embeds_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_array.append([float(item) for item in row]) 
    
    return string_array, float_array

   
def remove(speaker_name):
    try:
        speaker_names, speaker_embeds = load_stored_embeddings(speaker_names_file, speaker_embeds_file)
    except:
        print("Files do not exist")

    for i in range(0, len(speaker_names)):
        if speaker_names[i] == speaker_name:
            speaker_names.pop(i)
            speaker_embeds.pop(i)
            break

    store_stored_embeddings(speaker_names_file, speaker_embeds_file, speaker_embeds, speaker_names)

    
def train(audioFile, speakerName):
    global speaker_names
    global speaker_embeds
    try:
        speaker_names, speaker_embeds = load_stored_embeddings(speaker_names_file, speaker_embeds_file)
    except:
        print("Files do not exist")

    wav_fpath = Path("audio_data", audioFile)
    wav = preprocess_wav(wav_fpath)
    segments = [[0, 20]]
    speaker_names.append(speakerName)
    speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]
    encoder = VoiceEncoder("cpu")
    smth = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    speaker_embeds.append(smth[0])
    store_stored_embeddings(speaker_names_file, speaker_embeds_file, speaker_embeds, speaker_names)

    return 


def test(audioFile):
    global speaker_embeds, speaker_names
    try:
        speaker_names, speaker_embeds = load_stored_embeddings(speaker_names_file, speaker_embeds_file)
    except:
        print("No trained data exist")

    wav_fpath = Path("audio_data", audioFile)
    wav = preprocess_wav(wav_fpath)
    encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on cpu, this might take a while...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    # for i in stored_embeddings:
        # speaker_embeds = stored_embeddings[i]

    # speaker_embeds = [np.array(stored_embeddings.values())]
        # speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    # speaker_names = [np.array(stored_embeddings.keys())]
    # speaker_embeds = np.array(speaker_embeds)

    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                zip(speaker_names, speaker_embeds)}

    ## Run the interactive demo
    interactive_diarization(recognizer, similarity_dict, wav, wav_splits)
    

def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


def check_noise_level():
    duration = 1 
    threshold = 10
    while True:
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype=np.float32)
        sd.wait()

        rms = np.sqrt(np.mean(recording**2))
        print(f"Noise Level: {20 * np.log10(rms)} dB")

        if 20 * np.log10(rms) > threshold:
            print("It's too noisy!")
            decision = input("Do you want to start recording (R) or find a quieter environment (Q)? ").strip().lower()
            if decision == 'r':
                print("Recording started.")
                break
            elif decision == 'q':
                print("Please find a quieter environment.")
                break
            else:
                print("Invalid input. Please enter 'R' to record or 'Q' to find a quieter environment.")
            time.sleep(1)  # Check noise level every 1 second
        else: return


def capture_voice_input(output_file):
    flag = 1
    check_noise_level()
    audio = ''
    audio_path = Path("audio_data", output_file)
    while (not(keyboard.is_pressed('s'))):
            if flag:
                print("please press s to start recording or press l to leave recording")
                flag = 0
            if keyboard.is_pressed('l'):
                print("No audio recorder")
                return ''

    if keyboard.is_pressed('s'):
        with open(audio_path, "wb") as f:
            while True:
                with sr.Microphone() as source:
                    # recognizer.adjust_for_ambient_noise(source, duration=2)
                    print("Listening...")
                    try:
                        recognizer.pause_threshold = 100
                        audio, q = recognizer.listen(source, timeout=100)
                        text = convert_voice_to_text(audio)
                        print(text)
                    except sr.WaitTimeoutError:
                        print("Timeout occurred, please try again.")
                        if q :
                            break 
                        continue

                    if audio != '':
                        f.write(audio.get_wav_data())

                    if q:
                        break

               
                if keyboard.is_pressed('q'):
                    break
            
            
    # y, sr2 = librosa.load(audio_path)
    # # Apply energy-based VAD
    # energy = librosa.feature.rms(y=y)
    # threshold = np.mean(energy)
    # speech_segments = librosa.effects.split(y, top_db=threshold)

    # for seg in speech_segments:
    #     print("Speech detected from {:.2f}s to {:.2f}s".format(seg[0]/sr2, seg[1]/sr2))
    if audio == '':
        print("No audio recorder")
    else:
        print("audio saved")
    return audio


def convert_voice_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        text = ""
        print("Sorry, I didn't understand that.")
    except sr.RequestError as e:
        text = ""
        print("Error; {0}".format(e))
    return text


def process_voice_command(text):
    if "hello" in text.lower():
        print("Hello! How can I help you?")
    elif "goodbye" in text.lower():
        print("Goodbye! Have a great day!")
        return True
    

def main():
    end_program = False
    while not end_program:
        SpeakText("Hello, it's me back") 
        SpeakText("To train press t")
        for cnt in range (0, 100000):
            if keyboard.is_pressed('t'):
                SpeakText("please say your name clearly")
                speaker_name = convert_voice_to_text(capture_voice_input(output_file2))
                SpeakText("Read the following text")
                random_text = generate_random_text(7)
                print(random_text)
                audio = capture_voice_input(output_file2)    # capture a voice and save it in the specified file in audio_data folder
                train(output_file2, peaker_name)     #train on the voice file specified and add it to the database as speaker_name
                break

        SpeakText("To start conversation press c")
        for cnt in range (0, 100000):
            if keyboard.is_pressed('c'):
                print("hi")
                audio = capture_voice_input(output_file2)
                test(output_file2)  #test the specified file and present the result 
                break
        
        end_program = True



if __name__ == "__main__":
    active_device = get_active_physical_output_device()
    if active_device:
        print("Currently active physical output device:", active_device)
    else:
        print("No physical output device found.")


    active_device = get_active_physical_input_device()
    if active_device:
        print("Currently active physical input device:", active_device)
    else:
        print("No physical input device found.")

    main()


   