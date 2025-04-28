import fasttext

def predict_language(text, model): 
    try:
        if not isinstance(text, str) or len(text.strip()) < 2:
            return "Unknown"
        
        # Clean text (remove special chars, keep CJK + letters)
        clean_text = ''.join(c for c in text if c.isalpha() or c.isspace() or ord(c) > 127)
        if len(clean_text) < 2:
            return "Unknown"
        
        # Predict with FastText
        (lang,), (conf,) = model.predict(clean_text, k=1)
        lang = lang.replace("__label__", "")
        
        return lang.capitalize() if conf >= 0.7 else "Unknown"
    except Exception as e:
        print(f"Language detection error for '{text}': {str(e)}")
        return "Unknown"
    
if __name__ == "__main__":
    model_path = "lid.176.bin"
    model = fasttext.load_model(model_path)   

    test_cases = [
        ("YOASOBI 夜に駆ける", "Ja"),  # Japanese
        ("Jessie J Price Tag", "En"),  # English
        ("BTS Dynamite", "Ko"),        # Korean
        ("周杰倫 晴天", "Zh")           # Chinese
    ]

    for text, expected in test_cases:
        print(f"{text} → {predict_language(text, model)} (expected: {expected})")