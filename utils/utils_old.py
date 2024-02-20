from torchtext.vocab import build_vocab_from_iterator
import spacy

# Load the English model
eng = spacy.load('en_core_web_sm')

def load_dataset(data_path):
    data = []
    lines = open(data_path, 'r').read().strip().split('\n')
    for line in lines:
        temp = line.split('\t')
        qa = temp[1].split('?')
        
        answer = qa[2] if len(qa) == 3 else qa[1]

        data_sample = {'image_path': temp[0][:-2],
                    'question': qa[0] + '?',
                    'answer': answer.strip()}
        
        data.append(data_sample)

    return data

def get_tokens(data_iter):
    for sample in data_iter:
        question = sample['question']
        yield [token.text for token in eng.tokenizer(question)]

# build vocab
def build_vocab(data):
    vocab = build_vocab_from_iterator(
        get_tokens(data),
        min_freq=2,
        specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])
    
    return vocab


class Tokenizer(object):
    def __init__(self, vocab, max_sequence_length):
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length

    def __call__(self, question):
        tokens= [token.text for token in eng.tokenizer(question)]
        sequence = [self.vocab[token] for token in tokens]
        
        if len(sequence) < self.max_sequence_length:
            sequence += [self.vocab['<pad>']] * (self.max_sequence_length - len(sequence))
        else:
            sequence = sequence[:self.max_sequence_length]
            
        return sequence

    

if __name__ == "__main__":
    val_data = load_dataset('dataset/vaq2.0.TestImages.txt') 
    print(len(val_data))