import nltk
nltk.download('punkt')

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize

# -------------------------
# Word level extraction
# -------------------------
def extract_words(kws, doc_wordlist, size, docname):
    doc_flag = np.zeros(len(doc_wordlist))

    # go over each keyword
    for kw in kws:
        kw_phrase_len = len(kw.split())

        # go over the document (list of tokens)
        for j in range(len(doc_wordlist)-kw_phrase_len+1):
            doc_wds = ' '.join(doc_wordlist[j:(j+kw_phrase_len)])
            if doc_wds==kw: # a match is found at position j
                doc_flag[j] = 1
                for i in range(size):
                    right_idx = j+i+1
                    if right_idx <= len(doc_flag):
                        doc_flag[right_idx] = 1
                    left_idx = j-i-1
                    if left_idx>=0:
                        doc_flag[left_idx] = 1

    # go over each segment and save results
    doc_colloc = pd.DataFrame(columns=['docname','left','right','context'])
    num_segs = 0

    range_from = 0
    range_to = len(doc_flag)
    any_match = np.sum(doc_flag[range_from:range_to])
    while any_match>0:
        left_index = range_from + np.argmax(doc_flag[range_from:range_to]==1)
        if np.argmax(doc_flag[left_index:range_to]==0)==0:
            right_index = range_to # no more 0 at the end
        else:
            right_index = left_index + np.argmax(doc_flag[left_index:range_to]==0)

        # save this segment
        doc_colloc.at[num_segs,'docname'] = docname
        doc_colloc.at[num_segs,'left'] = left_index
        doc_colloc.at[num_segs,'right'] = right_index
        doc_colloc.at[num_segs,'context'] = ' '.join(doc_wordlist[left_index:right_index])
        num_segs = num_segs + 1

        # continue to search after
        range_from = right_index
        if range_from>=range_to:
            any_match = 0
        else:
            any_match = np.sum(doc_flag[range_from:range_to])
    return doc_colloc

# -------------------------
# Sent level extraction
# -------------------------
def extract_units(kws, doc_sentlist, docname):
    doc_colloc = pd.DataFrame(columns=['docname','context'])
    num_segs = 0
    for sent in doc_sentlist:
        wordlist = word_tokenize(sent.lower())
        break_flag = False

        # go over each keyword
        for kw in kws:
            kw_phrase_len = len(kw.split())

            # go over the document (list of tokens)
            for j in range(len(wordlist)-kw_phrase_len+1):
                doc_wds = ' '.join(wordlist[j:(j+kw_phrase_len)])
                if doc_wds==kw: # a match is found (at position j)
                    doc_colloc.at[num_segs,'docname'] = docname
                    doc_colloc.at[num_segs,'context'] = sent
                    num_segs = num_segs + 1
                    break_flag = True
                    break

            if break_flag:
                break

    # go over each segment and save results
    return doc_colloc

def extract_collocation(kws, data, level="word", size=4):
    data['context'] = ''
    docs = data['text'].to_list()
    if level=="word": # find context by nearby words
        wordlists = [word_tokenize(x.lower()) for x in docs]
        for i in range(data.shape[0]):
            docname = data.at[i,'docname']
            #print(docname)
            doc_wordlist = wordlists[i] # one document (as a list of tokens)
            doc_colloc = extract_words(kws, doc_wordlist, size, docname)
            data.at[i,'context'] = " ".join(doc_colloc['context'].to_list())

    if level[:4]=="sent": # find context by sentence
        sentlists = [sent_tokenize(x) for x in docs]
        for i in range(data.shape[0]):
            docname = data.at[i,'docname']
            #print(docname)
            doc_sentlist = sentlists[i] # one document (as a list of sentences)
            doc_colloc = extract_units(kws, doc_sentlist, docname)
            data.at[i,'context'] = " ".join(doc_colloc['context'].to_list())

    return data

if __name__ == "__main__":

    # read and clean text
    data = pd.read_csv("../data.csv")
    data['docname'] = pd.Series(["text" + str(i) for i in range(1, data.shape[0]+1)])

    # load the word list
    kws_df = pd.read_csv("../2.1 Collocation in R/kws_pretrained.txt", sep='\t', header=None)
    kws = kws_df.iloc[:,0].to_list()
    kws = list(set([x.lower() for x in kws]))
    print(kws)

    # merge results to original data frame
    word_results = extract_collocation(kws, data, level="word", size=4)
    word_results.to_csv("collocation_size4_results.csv", index=False)

    # alternative method
    sent_results = extract_collocation(kws, data, level="sent")
    sent_results.to_csv("collocation_sent_results.csv", index=False)
