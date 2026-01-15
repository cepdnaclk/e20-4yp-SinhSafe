import re
import nltk
from nltk.corpus import words

# Setup English Vocab
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class OfflineConverter:
    def __init__(self):
        # 1. English Dictionary
        self.english_vocab = set(words.words())
        self.english_vocab.update(['phone', 'gym', 'car', 'bus', 'laptop', 'internet', 'online', 'ok', 'set'])

        # 2. Transliteration Map (Singlish -> Sinhala)
        self.vowels = {
            'aa': 'ආ', 'a': 'අ', 'ii': 'ඊ', 'i': 'ඉ', 'uu': 'ඌ', 'u': 'උ', 
            'ee': 'ඊ', 'e': 'එ', 'ea': 'ඒ', 'oo': 'ඕ', 'o': 'ඔ', 
            'au': 'ඖ', 'ou': 'ඖ', 'ae': 'ඇ', 'aae': 'ඈ', 'ai': 'අයි'
        }
        self.consonants = {
            'ch': 'ච්', 'sh': 'ෂ්', 'gn': 'ඥ්', 'kn': 'ඤ්', 'th': 'ත්', 'dh': 'ද්', 
            'ph': 'ෆ්', 'kh': 'ඛ්', 'bh': 'භ්', 'k': 'ක්', 'g': 'ග්', 't': 'ට්', 
            'd': 'ඩ්', 'n': 'න්', 'p': 'ප්', 'b': 'බ්', 'm': 'ම්', 'y': 'යි', 
            'r': 'ර්', 'l': 'ල්', 'v': 'ව්', 'w': 'ව්', 's': 'ස්', 'h': 'හ්', 
            'f': 'ෆ්', 'j': 'ජ්'
        }
        self.modifiers = {
            'aa': 'ා', 'a': '', 'ii': 'ී', 'i': 'ි', 'ee': 'ී', 'uu': 'ූ', 'u': 'ු', 
            'ea': 'ේ', 'e': 'ෙ', 'oo': 'ෝ', 'o': 'ො', 'au': 'ෞ', 'ou': 'ෞ',
            'ae': 'ැ', 'aae': 'ෑ', 'ai': 'ෛ'
        }
        
        # Sort for Regex Priority
        self.sorted_vowels = sorted(self.vowels.keys(), key=len, reverse=True)
        self.sorted_consonants = sorted(self.consonants.keys(), key=len, reverse=True)
        self.sorted_modifiers = sorted(self.modifiers.keys(), key=len, reverse=True)

    def is_sinhala(self, word):
        """Checks if a word is already Sinhala (Unicode)"""
        return bool(re.search('[\u0D80-\u0DFF]', word))

    def remove_usernames(self, text):
        """Removes @mentions like @USER, @kamal from text."""
        # Ensure text is string
        text = str(text)
        # Regex to remove @ followed by word characters
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transliterate_word(self, text):
        res = text.lower()
        for c in self.sorted_consonants:
            c_base = self.consonants[c][:-1]
            for v in self.sorted_modifiers:
                if c + v in res:
                    res = res.replace(c + v, c_base + self.modifiers[v])
        for v in self.sorted_vowels:
            res = res.replace(v, self.vowels[v])
        for c in self.sorted_consonants:
            res = res.replace(c, self.consonants[c])
        return res

    def process_sentence(self, sentence):
        if not isinstance(sentence, str): return ""
        
        # --- 1. Clean Noise First (Remove @usernames) ---
        sentence = self.remove_usernames(sentence)
        
        tokens = sentence.split()
        processed_tokens = []

        for token in tokens:
            clean_token = re.sub(r'[^\w]', '', token).lower()
            
            # 1. Sinhala -> Keep
            if self.is_sinhala(token):
                processed_tokens.append(token)
            # 2. English -> Keep
            elif len(clean_token) > 2 and clean_token in self.english_vocab:
                processed_tokens.append(token)
            # 3. Numbers -> Keep
            elif not clean_token.isalpha():
                processed_tokens.append(token)
            # 4. Singlish -> Transliterate (Offline)
            else:
                converted = self.transliterate_word(clean_token)
                processed_tokens.append(converted)
                
        return " ".join(processed_tokens)

# --- Usage Example ---
if __name__ == "__main__":
    converter = OfflineConverter()
    # Test with a username
    test_text = "@kamal mama gedara yanawa phone eka genna"
    print(f"Original: {test_text}")
    print(f"Processed: {converter.process_sentence(test_text)}")