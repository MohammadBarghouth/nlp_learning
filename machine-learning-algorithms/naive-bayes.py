from functools import reduce
import re
from typing import Dict, Tuple, TypeVar
from utils.preprocessing import *
from utils.costum_types import *

import numpy as np

T = TypeVar("T")
Y = TypeVar("Y")

NUMBER = TypeVar("NUMBER", int, complex, float)

# General Help methods ---------------------------------

def unique_combinations(list_1: list[T], list_2: list[Y]) -> list[Tuple[T,Y]]:
    unique_combinations: list[Tuple[T,Y]] = []
 
    for i in range(len(list_1)):
        for j in range(len(list_2)):
            unique_combinations.append((list_1[i], list_2[j]))
    
    return unique_combinations

def multiply(l: list[NUMBER]) -> NUMBER:
    return reduce(lambda x, y: x*y,l)


# Naive Bias ---------------------------------

class Naive_Bias():
    labels: list[Label]
    documents: list[Document]
    vocabulary: Vocubulary
    label_documents: Dict[Label, list[Document]]
    
    _p_label: Dict[Label, float]
    _p_word_label: Dict[Tuple[str, Label], float]
    
    def __init__(self, documents: list[Document], labels: list[Label]):
        assert len(documents) == len(labels)
        
        self.labels = labels
        self.documents = documents
        self.vocabulary = create_vocabulary(self.documents)
        
        self.label_documents = {
            label: extract_documents_with_label(label, self.documents, self.labels)
            for label in set(self.labels)
        }
        
        self._p_label = {
            label: self.calculate_p_label(label)
            for label in self.labels
        }
        
        self._p_word_label = {
            (word, label): self.calculate_p_word_label(word, label)
            for (word, label) in unique_combinations(self.vocabulary, self.labels)
        }
        
    
    def calculate_p_word_label(self, word: str, label: Label):
        n_word = word_count_in_documents(word, self.label_documents[label]) # how much does our word appear in document that are labeled positively
        n = len(self.label_documents[label]) # number of all words in the positive case
            
        return (n_word + 1) / (n + len(self.vocabulary))
    
    def calculate_p_label(self, label: Label):
        return len(self.label_documents[label]) / len(self.documents)
    
    def p_label(self, label: Label):
        return self._p_label.get(label, 1)
    
    def p_word_label(self, word: str, label: Label):
        return self._p_word_label.get((word, label), 1)

    def p_doc_label(self, raw_text: str, label: Label):
        document = words(raw_text)
        return multiply([self.p_label(label)] + [self.p_word_label(word, label) for word in document])
         
    def predict_labels(self, raw_text: str):
        return [self.p_doc_label(raw_text, label) for label in set(self.labels)]

    
    def predict(self, raw_text: str):
        best_label_index = np.argmax(self.predict_labels(raw_text))
        return self.labels[best_label_index]

# use --------------------------------------------

def preprocess(raw_texts: list[str]):
    return Preprocessing(raw_texts)\
        .to_lower_case()\
        .remove_stop_words(2)\
        .build()

raw_texts = [
    "I love to eat pizza",
    "Cats are my favorite animal",
    "Programming is fun",
    "I enjoy hiking in the mountains",
    "Coffee keeps me awake",
    "I prefer reading books over watching TV",
    "I like to travel to different countries",
    "Dogs are loyal companions"
]

documents: list[Document] = preprocess(raw_texts)

labels: list[int] = [0, 1, 2, 0, 1, 2, 0, 1]

naive_bayes = Naive_Bias(labels=labels, documents=documents)

test_phrase = ""

print( naive_bayes.predict_labels(test_phrase), naive_bayes.predict(test_phrase))