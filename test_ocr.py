import ocr_engine

reader = ocr_engine.Reader()

text = reader.readtext('./0.png')

print(text)
