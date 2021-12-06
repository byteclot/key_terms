import collections
from nltk.corpus import stopwords
import string
import nltk
from lxml import etree
from collections import Counter
from nltk import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

tree = etree.parse('news.xml')
# tree = etree.parse('data.txt')

root = tree.getroot()

# for children in root.getchildren():
#     for c in children:
#         etree.dump(c.find("head"))

corpus = root.find("corpus")
news = corpus.findall("news")

punctuation_list = list(string.punctuation)
# print(punctuation_list)
stop_words = list(stopwords.words('english')) + ['ha', 'wa', 'u', 'a']
# print(stop_words)

lemmatizer = WordNetLemmatizer()
all_words = []
for n in news:
    value = n.findall("value")
    article_tokens = nltk.tokenize.word_tokenize(value[1].text.lower())
    article_tokens.sort(reverse=True)
    lemmatized_tokens = []
    for token in article_tokens:
        word = lemmatizer.lemmatize(token)
        if nltk.pos_tag([word])[0][1] == "NN":
            lemmatized_tokens.append(word)

    filtered_tokens = [token for token in lemmatized_tokens if
                       token not in stop_words and token not in punctuation_list and token not in ['ha', 'wa', 'u', 'a']]
    # print(filtered_tokens)
    all_words.append(" ".join(filtered_tokens))

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(all_words)
# print(tfidf_matrix.shape)
# print(tfidf_matrix[0])
# print(tfidf_matrix[1])
# print(tfidf_matrix.toarray())




# print(scores_dict)


# print(len(all_words))

#print(all_words)
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(all_words)
# feature_names = vectorizer.get_feature_names()
# df = pd.DataFrame(tfidf_matrix.toarray(), columns = vectorizer.get_feature_names())
# print(df)

# def get_ifidf_for_words(text):
#     tfidf_matrix= vectorizer.transform([text]).todense()
#     feature_index = tfidf_matrix[0,:].nonzero()[1]
#     tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
#     return dict(tfidf_scores)
#
# print(get_ifidf_for_words(all_words))

news_index = 0
for n in news:
    value = n.findall("value")
    head = f'{value[0].text}:'
    # article_tokens = nltk.tokenize.word_tokenize(value[1].text.lower())
    # article_tokens.sort(reverse=True)
    # lemmatized_tokens = []
    # for token in article_tokens:
    #     word = lemmatizer.lemmatize(token)
    #     if nltk.pos_tag([word])[0][1] == "NN":
    #         lemmatized_tokens.append(word)
    # fil_tok = [token for token in lemmatized_tokens if token not in stop_words and token not in punctuation_list and token not in ['ha', 'wa', 'u', 'a']]
    # #print(filtered_tokens)
    print(head)

    fil_tok = all_words[news_index].split()
    # print(fil_tok)

    my_df = pd.DataFrame(tfidf_matrix[news_index].T.todense(), index=vectorizer.get_feature_names_out(), columns=["TF-IDF"])
    my_df = my_df.sort_values('TF-IDF', ascending=False)
    #print(my_df.head(25))
    scores_dict = my_df.to_dict()['TF-IDF']

    sorted_dict = sorted(scores_dict.items(), key=lambda x: (x[1],x[0]), reverse=True)
    #print(sorted_dict)

    # df = pd.DataFrame(tfidf_matrix[news_index].toarray())
    # df2 = df.transpose().sort_values(by=0, ascending=False).reset_index()
    # terms = vectorizer.get_feature_names_out()
    # # print(terms)
    # top_words = []
    # for _i in range(0, 5):
    #     part = terms[int(df2.iloc[_i]['index'])]
    #     #string_to_print += part + ' '
    #     top_words.append(part)
    # #print(top_words)

    for i in range(5):
        print(sorted_dict[i][0], end=" ")
    print()
    # sprint(*sorted(top_words, reverse=True))


    # highest_scores = []
    # for k, v in scores_dict.items():
    #     if k in fil_tok:
    #         print(k, v)
    #         highest_scores.append(k)
    #         if len(highest_scores) == 10:
    #             break

    # final = []
    # for w in sorted(fil_tok, reverse=True):
    #     if w not in final and w in highest_scores:
    #         final.append(w)

    # print(*reversed(final))
#    print(*highest_scores)
    # my_dict = collections.Counter(fil_tok)
    # most_common_list = my_dict.most_common(5)
    # for i in range(5):
    #     print(most_common_list[i][0], end=' ')
    print()
    news_index += 1

