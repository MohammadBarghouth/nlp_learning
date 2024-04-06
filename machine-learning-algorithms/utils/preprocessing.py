from copy import deepcopy
import re
from typing import Dict, Self
from utils.costum_types import *

class Preprocessing:
    documents: list[Document]
    
    def __init__(self, raw_texts: list[str]) -> None:
        self.documents = create_documents(raw_texts)

    def to_lower_case(self) -> Self:
        self.documents = [[word.lower() for word in doc] for doc in self.documents]
        return self

    def remove_stop_words(self, K: int) -> Self:
        word_count: Dict[str, int] = {}
        
        for doc in self.documents:
            for word in doc:
                word_count[word] = word_count.get(word, 0) + 1
                
        stop_words = [word for word, count in word_count.items() if count >= K]
        
        for stop_word in stop_words:
            for doc in  self.documents:
                if stop_word in doc:
                    doc.remove(stop_word)
                
        return self
    
    def build(self):
        return self.documents
    
def words(document: str) -> Document:
    return re.findall(r'\b\w+\b', document)

def create_documents(documents: list[str]) -> list[Document]:
    return [words(doc) for doc in documents] 

def create_vocabulary(documents: list[Document]) -> list[str]:
    vocabulary: set[str] = set()
    
    for doc in documents:
        for word in doc:
            vocabulary.add(word)
            
    return list(vocabulary)

def word_count_in_document(word: str, document: Document) -> int:
    return sum([word == w for w in document])

def word_count_in_documents(word: str, documents: list[Document]) -> int:
    return sum([word_count_in_document(word, doc) for doc in documents])

def extract_documents_with_label(label: Label, documents: list[Document], labels: list[Label]):
    return [document for i, document in enumerate(documents) if labels[i] == label]
