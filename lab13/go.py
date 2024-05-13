import re
import nltk
from collections import Counter
import pymorphy2

text = open('txt', 'r').read()
nltk.download('stopwords')

#токенизация сплитом
print(text.split())

#токенизация при помощи регулярных выражений
tokens = re.findall(r"\w+", text)
print(tokens)

#токенизация при помощи NLTK
print(nltk.word_tokenize(text))

'''Токенизация сплитом является не совсем токенизацией, данный способ разбивает текст без учета пунктуации. Способ регулярных выражений и 
способ с использованием модуля nltk разбивают текст с учетом пунктуации и оставляет чистые слова.'''

#кол-во слов в тексте разными функциями
print(len(text))

print(Counter(text))

'''Результаты отличные, так как Counter подсчитывает кол-во разных символов в текста, также с учетом знаков препинания, и выдает это в формате словаря. 
Функия len считает кол-во всех символов в тексте и возвращает целое число.'''

#очистка при помощи isalpha
new_text = [i for i in text.split() if i.isalpha()]
print(new_text)

#очистка от стоп-слов
filtered_words = [word for word in nltk.word_tokenize(text) if word not in nltk.corpus.stopwords.words('english')]
print(filtered_words)

#стемминг при помощи nltk
stemmer = nltk.stem.PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in nltk.word_tokenize(text)]
stemmed_text = " ".join(stemmed_tokens)
print(stemmed_text)

#лемматизация при помощи pymorphy2
morph = pymorphy2.MorphAnalyzer()
lemmatized_text = " ".join([morph.parse(j)[0].normal_form for j in text.split()])
print(lemmatized_text)

'''Лемматизация предосталвяет нам слова в их начальной форме, в то время как стемминг лишь отбрысвает буквы в слове, оставляя основу'''