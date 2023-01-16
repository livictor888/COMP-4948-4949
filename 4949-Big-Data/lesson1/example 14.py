import speech_recognition as sr
r   = sr.Recognizer()
mic = sr.Microphone()

text = ""
while text != "stop":

    with mic as source:
        # Add the following line to filter out background noise.
        # r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    #output = r.recognize_google(audio)
    text   = r.recognize_google(audio, language='en-IN')

print(text)
