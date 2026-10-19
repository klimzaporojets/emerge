import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words

# Ensure the required NLTK corpora are present before we access them at import
# time (any module that imports this one must not depend on a prior download).
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)

# Load the list of English words
english_words = set(words.words())


def calculate_english_word_percentage(text):
    # Tokenize the input text
    tokens = word_tokenize(text)

    # Filter out non-English words
    english_word_count = sum(1 for word in tokens if word.lower() in english_words)

    # Calculate the percentage
    total_words = len(tokens)
    percentage = (english_word_count / total_words) * 100 if total_words > 0 else 0

    return percentage
