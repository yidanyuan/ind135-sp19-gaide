import speech_recognition as sr
r = sr.Recognizer()
harvard = sr.AudioFile('english.wav')

with harvard as source:
    audio = r.record(source)
    txt = r.recognize_google(audio)
    print(txt)