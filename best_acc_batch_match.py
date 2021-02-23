import ast
import os
import operator
import math
import json
import argparse
import codecs
import string
import nltk
import jieba
import time

def read_data_seperated(path):

    with codecs.open(path, 'r', encoding='utf-8') as f1:
        result_dict = json.load(f1, encoding='utf-8')

    data_dict = {}
    new_data_dict = {}

    for item in result_dict.items():
        data_list = []
        position_list = []
        for position_item in ast.literal_eval(item[1]):
            data_list.append(position_item[1])
            position_list.append(position_item[0])
        data_dict[str(item[0])] = (data_list, position_list)

    for item in data_dict.items():

        new_data_list = []
        new_position_list = []
        position_list = item[1][1]

        token_list = [[tokenize(str)] for str in item[1][0]]

        for token, position in zip(token_list, position_list):

            for token_item in token[0]:
                new_data_list.append(token_item)
                new_position_list.append(position)

        new_data_dict[str(item[0])] = (new_data_list, new_position_list)

    return new_data_dict

def tokenize(s_data):

    translator = str.maketrans('', '', string.punctuation)
    modified_string = s_data.translate(translator)
    # modified_string = s_data
    # for chinese string
    modified_string = jieba.cut(modified_string)
    tokenized_str = [item for item in modified_string if item is not " "]
    # for English
    #tokenized_str = nltk.word_tokenize(modified_string)

    return tokenized_str

def get_vocabulary(data):

    tokens = []

    for key, content in data.items():
        for item in content[0]:
            tokens.append(item)

    # fdist = nltk.FreqDist(tokens)
    # return list(fdist.keys())

    return list(set(tokens))

def generate_inverted_index(data):

    all_words = get_vocabulary(data)

    index = {}

    for word in all_words:
        for doc_num, tokens in data.items():
            if word in tokens[0]:
                if word in index.keys():
                    index[word].append((doc_num, tokens[1][tokens[0].index(word)]))
                else:
                    index[word] = [(doc_num, tokens[1][tokens[0].index(word)])]

    return index

def calculate_idf(data):

    idf_score = {}

    data_len = len(data)

    all_words = get_vocabulary(data)

    for word in all_words:
        word_count = 0
        for key, token_list in data.items():
            if word in token_list[0]:
                word_count += 1

        idf_score[word] = math.log10(data_len / word_count)


    return idf_score

def calculate_tf(tokens):

    tf_scores = {}

    for token in tokens:
        tf_scores[token] = tokens.count(token)
        tf_scores[token] = 1

    return tf_scores


def calculate_tfidf(data, idf_score):

    tf_idf_scores = {}
    tf_scores = {}

    for key, value in data.items():
        tf_scores[key] = calculate_tf(value[0])

    for doc_num, tf_score in tf_scores.items():
        for token, score in tf_score.items():
            tf = score
            idf = idf_score[token]
            tf_score[token] = tf * idf
        tf_idf_scores[str(doc_num)] = tf_score

    return tf_idf_scores

def calculate_tf_q(tokens):

    tf_scores = {}

    for token in tokens:
        tf_scores[token] = 1

    return tf_scores


def calculate_tfidf_queries(queries, idf_score):

    q_tf_idf_scores = {}
    tf_scores = {}
    for key, value in queries.items():
        tf_scores[key] = calculate_tf_q(value[0])

    for key, tf_score in tf_scores.items():
        for token, score in tf_score.items():
            tf = score
            if token in idf_score.keys():
                idf = idf_score[token]
            else:
                idf = 0
            tf_score[token] = tf * idf
        q_tf_idf_scores[str(key)] = tf_score

    return q_tf_idf_scores


def save_dict(dict_data, name):

    with codecs.open(name + ".json", 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, ensure_ascii=False)

def read_dict(name):

    with codecs.open(name, 'r', encoding='utf-8') as f1:
        result_dict = json.load(f1, encoding='utf-8')

    return result_dict



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", default='./data/scanned_result.json')
    parser.add_argument("--query_path", default='./data/camera_result.json')
    args = parser.parse_args()

    data = read_data_seperated(args.content_path)

    queries = read_data_seperated(args.query_path)
    # inverted_index = generate_inverted_index(data)
    # idf_scores = calculate_idf(data)
    # scores = calculate_tfidf(data, idf_scores)

    if os.path.exists("inverted_index.json"):
        print("load inverted index")
        inverted_index = read_dict("inverted_index.json")
    else:
        print("create inverted index")
        inverted_index = generate_inverted_index(data)
        save_dict(inverted_index, "inverted_index")

    if os.path.exists("idf_scores.json"):
        print("load idf scores")
        idf_scores = read_dict("idf_scores.json")
    else:
        print("create idf")
        idf_scores = calculate_idf(data)
        save_dict(idf_scores, "idf_scores")

    if os.path.exists("scores.json"):
        print("load tf-idf scores")
        scores = read_dict("scores.json")
    else:
        print("tf-idf")
        scores = calculate_tfidf(data, idf_scores)
        save_dict(scores, "scores")

    query_scores = calculate_tfidf_queries(queries, idf_scores)
    query_docs = {}
    for key, value in queries.items():
        time_start=time.time()
        doc_sim = {}

        for term in value[0]:
            if term in inverted_index.keys():
                docs_list = inverted_index[term]
                p1, p2, p3, p4 = value[1][value[0].index(term)]
                # print(f'{p1}-{p2}-{p3}-{p4}')
                average_h = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
                docs = []
                for item in docs_list:
                    c_p1, c_p2, c_p3, c_p4 = item[1]
                    # print(item[1])
                    # print(f'{c_p1}-{c_p2}-{c_p3}-{c_p4}')
                    c_average_h = (c_p1[1] + c_p2[1] + c_p3[1] + c_p4[1]) / 4
                    # print(c_average_h)
                    #print(c_average_h)
                    #print(average_h)
                    # if abs(c_average_h - average_h) > 1:
                    #     print(f'{p1}-{p2}-{p3}-{p4}')
                    #     print(f'{c_p1}-{c_p2}-{c_p3}-{c_p4}')

                    # 0.07 has 0.979 acc
                    # 0.075 has 0.981 acc
                    # 0.08 has 0.983 acc
                    # 0.09 has 0.983 acc
                    # 0.1 has 0.981 acc
                    # should manually changed according to the dataset
                    if abs(c_average_h - average_h) < 0.08:
                        docs.append(item[0])
                    # docs.append(item[0])

                for doc in docs:

                    # doc = str(doc)
                    doc_score = scores[str(doc)][term]
                    doc_length = math.sqrt(sum(x ** 2 for x in scores[str(doc)].values()))
                    query_score = query_scores[str(key)][term]
                    query_length = math.sqrt(sum(x ** 2 for x in query_scores[str(key)].values()))
                    cosine_sim = (doc_score * query_score) / (doc_length * query_length)
                    if doc in doc_sim.keys():
                        doc_sim[doc] += cosine_sim
                    else:
                        doc_sim[doc] = cosine_sim

                    '''
                    if str(doc) == '187':
                        print(f'query doc num: {key}')
                        print(f'term: {term}', end='  |')
                        print(f'doc_num - doc_score: {doc} - {doc_score}', end=' | ')
                        print(f'query_score: {query_score}', end='  |')
                        print(f'cosine_sim: {cosine_sim}', end='  |')
                        print()
                    '''

        ranked = sorted(doc_sim.items(), key=operator.itemgetter(1), reverse=True)

        query_docs[key] = ranked
        time_end=time.time()
        print('time cost',time_end-time_start,'s')

    #print(query_docs)
    count = 0
    total_num = len(query_docs)
    for key, value in query_docs.items():
        if value != []:
            matched_id = value[0][0]
        else:
            matched_id = '99999'
        if int(key) != int(matched_id):
            print(f"{key} mismatched with {matched_id}")
        else:
            count += 1

    print(f"acc is {count/total_num:.3f}")
