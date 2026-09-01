"""This script exports modules for on-the-fly augmentation and can also be run to generate a csv file of augmented data."""

import os
import random

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords, wordnet
from torch import nn
from tqdm import tqdm
from transformers import (
    FSMTForConditionalGeneration,
    FSMTTokenizer,
    pipeline,
)

from twitter import util

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    wordnet.synsets('dog')
except LookupError:
    nltk.download('wordnet')


################################
#                              #
#       backtranslation        #
#                              #
################################

class BackTranslation(nn.Module):
    """
    Backtranslation using the facebook/wmt19-en-de and facebook/wmt19-de-en models
    Based on: https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb

    Args:
        temperature: temperature for generation
        device: device to use
    """

    def __init__(self, temperature: float = 1.0, device='cuda'):
        super().__init__()
        en_de_name = "facebook/wmt19-en-de"
        de_en_name = "facebook/wmt19-de-en"

        self.device = device
        self.temperature = temperature

        self.fwd_tokenizer = FSMTTokenizer.from_pretrained(en_de_name)
        self.fwd_translator = FSMTForConditionalGeneration.from_pretrained(en_de_name).eval().to(self.device)
        self.bwd_tokenizer = FSMTTokenizer.from_pretrained(de_en_name)
        self.bwd_translator = FSMTForConditionalGeneration.from_pretrained(de_en_name).eval().to(self.device)

    def forward(self, batch: tuple[list[str], dict[str, torch.Tensor]]) -> list[str]:
        input_texts = batch[0]

        # forward translation
        fwd_tokens = self.fwd_tokenizer.batch_encode_plus(input_texts, return_tensors="pt", padding=True)
        fwd_tokens = {k: v.to(self.device) for k, v in fwd_tokens.items()}
        fwd_output = self.fwd_translator.generate(**fwd_tokens)
        fwd_translation = self.fwd_tokenizer.batch_decode(fwd_output, skip_special_tokens=True)

        # backward translation
        bwd_tokens = self.bwd_tokenizer.batch_encode_plus(fwd_translation, return_tensors="pt", padding=True)
        bwd_tokens = {k: v.to(self.device) for k, v in bwd_tokens.items()}
        bwd_output = self.bwd_translator.generate(**bwd_tokens, do_sample=True, top_k=0, temperature=self.temperature)
        bwd_translation = self.bwd_tokenizer.batch_decode(bwd_output, skip_special_tokens=True)

        return bwd_translation


################################
#                              #
#       random insertion       #
#                              #
################################

class RandomInsertion(nn.Module):

    def __init__(self, n: int = 5):
        super().__init__()
        self.n = n
        self.unmasker = pipeline('fill-mask', model='bert-base-cased', device=0)

    def forward(self, input_texts: list[str]):
        augmented_texts = input_texts
        for i in range(self.n):
            texts_with_mask = []
            for text in augmented_texts:
                tokens = text.split()
                mask_idx = random.randint(1, max(1, len(tokens) - 2))

                tokens_with_mask = tokens[:mask_idx] + ['[MASK]'] + tokens[mask_idx:]
                text_with_mask = ' '.join(tokens_with_mask)

                texts_with_mask.append(text_with_mask)

            outputs = self.unmasker(texts_with_mask)
            augmented_texts = [random.choice(output)['sequence'] for output in outputs]
        return augmented_texts


################################
#                              #
#       random replacement     #
#                              #
################################

class RandomReplacement(nn.Module):

    def __init__(self, n: int = 5):
        super().__init__()
        self.unmasker = pipeline('fill-mask', model='bert-base-cased', device=0)
        self.n = n

    def forward(self, input_texts: str):
        augmented_texts = input_texts
        for i in range(self.n):
            masked_texts = []
            for input_text in input_texts:
                tokens = input_text.split()

                replacement_idx = random.randint(1, max(1, len(tokens) - 1))
                orig_word = tokens[replacement_idx]
                tokens_with_mask = tokens.copy()
                tokens_with_mask[replacement_idx] = '[MASK]'
                masked_text = ' '.join(tokens_with_mask)

                masked_texts.append(masked_text)

            outputs = self.unmasker(masked_texts)
            augmented_texts = []
            for output in outputs:
                augmented_text = random.choice(output)
                if augmented_text['token_str'] != orig_word:
                    augmented_texts.append(augmented_text['sequence'])

        return augmented_texts


################################
#                              #
#       text generation        #
#                              #
################################

class TextGeneration(nn.Module):

    def __init__(self, num_new_words: int = 5, num_return_sequences: int = 5):
        super().__init__()
        self.generator = pipeline('text-generation', model='gpt2', device=0)
        self.num_new_words = num_new_words
        self.num_return_sequences = num_return_sequences

    def forward(self, input_text: str):
        input_length = len(input_text.split())
        output_length = input_length + self.num_new_words
        if input_length > self.num_new_words:
            input_text = ' '.join(input_text.split()[:-self.num_new_words])
        gpt_output = self.generator(input_text, max_length=output_length, num_return_sequences=self.num_return_sequences)
        return gpt_output[0]['generated_text']


################################
#                              #
#             EDA              #
#                              #
################################

# based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py

def get_synonyms(word: str) -> list[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


class EDARandomDeletion(nn.Module):

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, text: str):
        new_text = []
        for word in text.split():
            r = random.uniform(0, 1)
            if r > self.p:
                new_text.append(word)
        return ' '.join(new_text)


class EDARandomSwap(nn.Module):

    def __init__(self, n: int = 2):
        super().__init__()
        self.n = n

    def forward(self, text: str):
        new_text = text.split()
        for i in range(self.n):
            idx1 = random.randint(0, len(new_text) - 1)
            idx2 = random.randint(0, len(new_text) - 1)
            new_text[idx1], new_text[idx2] = new_text[idx2], new_text[idx1]
        return ' '.join(new_text)


class EDASynonymReplacement(nn.Module):

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, words: str):
        new_words = words.split(' ')
        random_word_list = list(set([word for word in words.split(' ') if word not in stopwords.words('english')]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= self.n:  # only replace up to n words
                break

        # this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        new_sentence = ' '.join(new_words)

        return new_sentence


class EDARandomInsertion(nn.Module):

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, words: str):
        new_words = words.split(' ')
        for _ in range(self.n):
            self.add_word(new_words)
        return ' '.join(new_words)

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)


class EDA(nn.Module):

    def __init__(self,
                 alpha_sr: float = 0.1,
                 alpha_ri: float = 0.0,
                 alpha_rs: float = 0.0,
                 p_rd: float = 0.0):
        super().__init__()
        self.alpha_sr = alpha_sr  # synonym replacement
        self.alpha_ri = alpha_ri  # random insertion
        self.alpha_rs = alpha_rs  # random swap
        self.p_rd = p_rd  # random deletion

    def forward(self, text: str):
        num_words = len(text.split())
        augmentations = ["none"]
        if self.alpha_sr > 0:
            n_sr = max(1, int(self.alpha_sr * num_words))
            augmentations.append('sr')
        if self.alpha_ri > 0:
            n_ri = max(1, int(self.alpha_ri * num_words))
            augmentations.append('ri')
        if self.alpha_rs > 0:
            n_rs = max(1, int(self.alpha_rs * num_words))
            augmentations.append('rs')
        if self.p_rd > 0:
            augmentations.append('rd')
        aug = random.choice(augmentations)

        if aug == 'sr':
            return EDASynonymReplacement(n=n_sr)(text)
        elif aug == 'ri':
            return EDARandomInsertion(n=n_ri)(text)
        elif aug == 'rs':
            return EDARandomSwap(n=n_rs)(text)
        elif aug == 'rd':
            return EDARandomDeletion(p=self.p_rd)(text)

        return text


if __name__ == '__main__':
    import argparse

    from twitter import data

    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation', type=str, default='random_insertion', help='augmentation method')
    parser.add_argument('--output_dir', type=str, default='data/augmentation/classification', help='output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='number of augmented samples per sample')
    parser.add_argument('--buffer_size', type=int, default=1000, help='buffer size for writing to csv')
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature for backtranslation')
    args = parser.parse_args()

    ds = data.TextDataset(root_dir="data/splits", split="train", preprocessing=util.TweetNormalizer())
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    transform = None
    if args.augmentation == 'backtranslation':
        transform = BackTranslation(temperature=args.temperature, device='cuda')
    elif args.augmentation == 'random_insertion':
        transform = RandomInsertion()
    elif args.augmentation == 'random_replacement':
        transform = RandomReplacement()
    elif args.augmentation == 'text_generation':
        transform = TextGeneration()
    elif args.augmentation == 'eda_random_deletion':
        transform = EDARandomDeletion()
    elif args.augmentation == 'eda_random_swap':
        transform = EDARandomSwap()
    elif args.augmentation == 'eda_synonym_replacement':
        transform = EDASynonymReplacement()
    elif args.augmentation == 'eda_random_insertion':
        transform = EDARandomInsertion()
    else:
        raise ValueError('augmentation method not supported')

    augmented_data = {'text': [], 'sentiment': [], 'score_compound': []}
    for batch in tqdm(dl):
        for i in range(args.num_samples):
            augmented_data['text'].extend(transform(batch).cpu())
            augmented_data['sentiment'].extend([x.item() for x in batch[1]['sentiment']])
            augmented_data['score_compound'].extend([x.item() for x in batch[1]['score_compound']])

        # write in buffers of 1000 to avoid memory issues
        if len(augmented_data['text']) >= args.buffer_size:
            print(os.path.join(args.output_dir, args.augmentation + '.csv'))
            pd.DataFrame(augmented_data).to_csv(
                os.path.join(args.output_dir, args.augmentation + '.csv'),
                index=False,
                mode='a',
                header=False
            )
            augmented_data = {'text': [], 'sentiment': [], 'score_compound': []}
