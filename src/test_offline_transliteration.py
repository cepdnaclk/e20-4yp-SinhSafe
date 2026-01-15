# Import the class from your file
from offline_transliteration import OfflineConverter 

def test_logic():
    converter = OfflineConverter()
    
    test_cases = [
        ("mama phone eka gatta ගෙදර", "මම phone"),                # Pure Singlish
        ("phone", "phone"),            # English word (Should stay English)
        ("gedara", "ගෙදර"),            # Pure Singlish
        ("123", "123"),                # Numbers
        ("mama phone eka gaththa", "මම phone එක ගත්ත"), # Mixed Sentence
    ]
    
    print("--- Testing Offline Converter ---")
    for input_text, expected_partial in test_cases:
        result = converter.process_sentence(input_text)
        print(f"Input: {input_text:<25} | Output: {result}")

if __name__ == "__main__":
    test_logic()