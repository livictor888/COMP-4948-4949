
import speech_recognition as sr
PATH = "C:\\datasets\\audiosamples1\\kramer_3.wav"

r = sr.Recognizer()

jackhammer = sr.AudioFile(PATH)
with jackhammer as source:
    audio = r.record(source, offset=1.5, duration=1.5)
    output = r.recognize_google(audio)
    print(output)
