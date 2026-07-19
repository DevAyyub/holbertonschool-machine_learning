#!/usr/bin/env python3
"""Module containing the Dataset class for machine translation"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Initializes the dataset and tokenizers"""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence

        Returns:
            tokenizer_pt: the Portuguese tokenizer
            tokenizer_en: the English tokenizer
        """
        base_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def get_pt():
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in pt]

        def get_en():
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [sentence.decode('utf-8') for sentence in en]

        tokenizer_pt = base_pt.train_new_from_iterator(
            get_pt(), vocab_size=2**13
        )
        tokenizer_en = base_en.train_new_from_iterator(
            get_en(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens: list containing the Portuguese tokens
            en_tokens: list containing the English tokens
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        pt_vocab = self.tokenizer_pt.vocab_size
        en_vocab = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab] + pt_tokens + [pt_vocab + 1]
        en_tokens = [en_vocab] + en_tokens + [en_vocab + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            result_pt: tf.Tensor containing the encoded Portuguese tokens
            result_en: tf.Tensor containing the encoded English tokens
        """
        result_pt, result_en = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
