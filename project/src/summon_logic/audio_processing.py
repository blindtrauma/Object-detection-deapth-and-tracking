import re
import speech_recognition as sr


def process_file_command(audio):

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        print("completed listening")
        try:
            text = recognizer.recognize_google(audio)
            print(text)
            print(f"Detected speech: {text}")
            return text

        except sr.UnknownValueError:
            print("None, error encountered")
        except sr.RequestError as e:
            print(f"None, eroor encountered; {e}")



import pyttsx3
import pyttsx3

def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for index, voice in enumerate(voices):
        print(f"Voice {index}: {voice.name} ({voice.languages})")

list_voices()


def text_to_speech(text, voice_index = 2):

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_index].id)

    rate = engine.getProperty('rate')   
    volume = engine.getProperty('volume') 
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()


text_to_speech('this image analyses a screwdriver')



import re
import speech_recognition as sr

def process_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        print("Completed listening")

    try:
        text = recognizer.recognize_google(audio)
        print(f"Detected speech: {text}")
        return text
    except sr.UnknownValueError:
        print("Error: Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Error: Could not request results; {e}")
        return None









