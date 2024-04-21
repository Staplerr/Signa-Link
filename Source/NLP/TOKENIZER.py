import spacy_thai
from thai_tokenizer import Tokenizer

nlp = spacy_thai.load()

word_dict = Tokenizer().__dict__['pairs']
thai_words = []
new_dict = {}

for i,j in enumerate(word_dict):
    new_key = ''.join(j)
    thai_words.append(new_key)
    new_dict[new_key] = i
    new_dict[i] = new_key

def separate_word( target):
        # Process the sentence with spaCy
        doc = nlp(target)

        # Reorder words based on their positions
        ordered_words = [token.text for token in doc]

        return ordered_words

class Tokenizer():

    def __init__(self) -> None:
        self.vocab = new_dict
        self.word = thai_words
        self.n_vocab = len(thai_words)

    def encode(self, target : str) -> list:
        token = []
        result = separate_word(target)
        try:
            for i in result:
                token.append(self.vocab[i])
        except KeyError as e:
            self.add_word(e.args[0])
            index = result.index(e.args[0])
            for i in result[index:]:
                token.append(self.vocab[i])
                print()
        return token

    def decode(self, target : list) -> str:
        word = []
        for i in target:
            try:
                word.append(self.vocab[i])
            except KeyError as e:
                self.add_word(e.args[0])
                
        return ''.join(word)

    def add_word(self, new_word):
        self.n_vocab += 1
        self.word.append(new_word)
        self.vocab[new_word] = self.n_vocab
        self.vocab[self.n_vocab] = new_word

        print(f'the word {new_word} has been added')

    def encode_list(self, lst : list) -> list:
        encoded = []
        for i in lst:
            try:
                encoded.append(self.encode(i))
            except KeyError as e:
                self.add_word(e.args[0])
                print(f'Error raised the word {e.args[0]} is not in the dict.')
                encoded.append(self.encode(i))
        return encoded

    def decode_list(self, lst : list) -> list:
        decoded = []
        for i in lst:
            try:
                decoded.append(self.decode(i))
            except KeyError as e:
                self.add_word(e.args[0])
                print(f'Error raised the word {e.args[0]} is not in the dict.')
                decoded.append(self.decode(i))
        return decoded