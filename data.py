import unicodedata
import re
import io

import tensorflow as tf

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    #w = unicode_to_ascii(w.lower().strip())
    w = w.lower()

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, ".", "?", "!", ",")
    w_punct = re.sub(r"^[a-z?.!,]\s+", " ", w)

  # replacing everything with space except (a-z)
    w = re.sub(r"^[a-z]\s+", " ", w)

  #remove multiple spaces
    w = re.sub(' +', ' ', w)
    w_punct = re.sub(' +', ' ', w_punct)

    w = w.strip()
    w_punct = w_punct.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    w_punct = '<start> ' + w_punct + ' <end>'
    return [w, w_punct]

# 1. Clean the sentences
# 2. Return sentence pairs in the format: [sentence without punct, sentence with punct]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [preprocess_sentence(l) for l in lines[:num_examples]]

    return zip(*word_pairs)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))
