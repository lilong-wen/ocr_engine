import ocr_engine

reader = ocr_engine.Reader()

text = reader.readtext('./0.png', detail=0, paragraph=True)

print(text)
