import re
import nltk
from collections import Counter
import pymorphy2
import matplotlib.pyplot as plt

text = open('txt', 'r').read()
nltk.download('stopwords')

#токенизация сплитом
print(text.split())

#токенизация при помощи регулярных выражений
tokens = re.findall(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", text)
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
filtered_words = [word for word in nltk.word_tokenize(text) if word not in nltk.corpus.stopwords.words('russian')]
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

#график наиболее частотных слов
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_words]
word_counts = Counter(lemmatized_tokens)
most_frequent_words = word_counts.most_common(20)

plt.barh([word for word, count in most_frequent_words], [count for word, count in most_frequent_words])
plt.title('График наиболее частотных лемматизированных слов')
plt.ylabel('Слово')
plt.xlabel('Частота')
plt.show()

#построить dispersion plot
bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(nltk.word_tokenize(text))
bigrams = bigram_finder.nbest(nltk.metrics.BigramAssocMeasures.likelihood_ratio, 10)

trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(nltk.word_tokenize(text))
trigrams = trigram_finder.nbest(nltk.metrics.TrigramAssocMeasures.likelihood_ratio, 10)

plt.scatter([t[1] for t in bigrams], [t[0] for t in bigrams])
plt.xlabel("О вероятность")
plt.ylabel("Значение коллокации")
plt.title("График рассеяния для биграмм")
plt.show()

plt.scatter([t[1] for t in trigrams], [t[0] for t in trigrams])
plt.xlabel("О вероятность")
plt.ylabel("Значение коллокации")
plt.title("График рассеяния для триграмм")
plt.show()

#грамматические характеристики
morph_1 = pymorphy2.MorphAnalyzer()
parsed = [morph.parse(word)[0] for word in text.split()]
for w in parsed:
    print(f"{w.word}: {w.tag}")
    
#график частотности частей речи
pos_tags = [nltk.pos_tag([word])[0][1] for word in text.split()]   # Разделить текст по словам и посчитать частотность частей речи
pos_counts = Counter(pos_tags)
pos_dict = list(pos_counts.items())   # Создать список с частями речи и их количеством

plt.bar([x[0] for x in pos_dict], [x[1] for x in pos_dict])
plt.xlabel("Часть речи")
plt.ylabel("Частота")
plt.title("Распределение частотности частей речи")
plt.xticks(rotation=45)
plt.show()

#тестировка методов similar, common_contexts и collocations
# Нахождение похожих слов
print("Подобные слова:")
for sim in nltk.text.Text.similar('book', text):
    print(f" - {sim}")
    
# Нахождение общих контекстов
print("\nОбщие контексты:")
for context in nltk.text.Text.common_contexts(['book', 'novel'], text):
    print(f" - {context}")
    
# Нахождение коллокаций
print("\nКоллокации:")
finder = nltk.collocations.BigramCollocationFinder.from_words(text)
finder.apply_freq_filter(3)
for collocation in finder.nbest(10):
    print(f" - {collocation}")
    
'''Similar: для поиска синонимов, антонимов и других семантически сходных слов; для улучшения классификации текста и извлечения сущностей.
   Common_contexts: для поиска слов, которые часто встречаются вместе в одном контексте; для выявления коллокаций; для улучшения производительности машинного обучения и обработки естественного языка.
   Collocations: Для обнаружения устойчивых сочетаний слов, которые часто встречаются вместе; для улучшения качества перевода.
'''
