# ---------------
# Building corpus 
# ---------------

import pandas as pd
import numpy as np

# read and clean text
data = pd.read_csv("../2.2 Collocation in Python/collocation_size4_results.csv")

docs_HighTech = data.loc[data["category"]=='HighTech', 'context'].fillna("").to_list()
docs_NonHighT = data.loc[data["category"]=='NonHighTech', 'context'].fillna("").to_list()

# remove stopwords
from gensim.parsing.preprocessing import STOPWORDS

# load the word list
kws = ["innovation", "innovations", "innovate", "innovates", \
       "innovative", "innovating", "technological innovation", \
       "technological innovations", "continually innovating", \
       "cutting edge", "innovators", "continually innovate", \
       "technology", "technologies", "innovated", \
       "continually innovates", "constantly innovating", \
       "intrapreneurship", "continuously innovating", \
       "constantly innovates", "innovator", "advancements", \
       "technological advancement", "technological advancements", \
       "entrepreneurial mindset", "technologic innovation", \
       "technological breakthroughs", "technological", \
       "commercialize", "revolutionizing", \
       "operational excellence", "technological advances", \
       "creativity", "entrepreneurial spirit", \
       "constantly striving", "innovativeness", \
       "continuously innovates", "technological developments", \
       "pioneering", "inventions", "continually strive", \
       "customer centric", "revolutionize", "entrepreneurship", \
       "inventive", "excellence", "operationally efficient", \
       "adapting", "reinvent", "rethink", "creative", "pioneer", \
       "entrepreneurs", "differentiators", "modernize", "evolve", \
       "invent", "developing", "differentiated", \
       "entrepreneurial", "breakthroughs", "ingenuity", "adapt", \
       "novel", "refining", "imagination"]
kws = list(set([x.lower() for x in kws]))

my_stopwords = list(STOPWORDS) + kws 

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation, \
    strip_numeric, strip_multiple_whitespaces, strip_short

def clean_text(docs):
    docs = [strip_punctuation(x) for x in docs]
    docs = [strip_numeric(x) for x in docs]
    docs = [strip_multiple_whitespaces(x) for x in docs]
    docs = [strip_short(x, minsize=2) for x in docs]
    docs = [remove_stopwords(x.lower(), my_stopwords) for x in docs]
    corpus = [x.split() for x in docs]
    return corpus

docs_HighTech = clean_text(docs_HighTech)
docs_NonHighT = clean_text(docs_NonHighT)

# create dictionary and filter words
id2word = corpora.Dictionary(docs_HighTech + docs_NonHighT)
corpus_HighTech = [id2word.doc2bow(text) for text in docs_HighTech]
corpus_NonHighT = [id2word.doc2bow(text) for text in docs_NonHighT]

# ---------------
# Determining the number of topics 
# ---------------

import gensim
import gensim.corpora as corpora

def find_perp(corpus_train,K_seq,seed_id):
    log_perp_values = [np.nan]*len(K_seq)
    for i in range(len(num_topics_range)):

        # train the model
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus_train, id2word=id2word, 
            num_topics=num_topics_range[i], random_state=seed_id
        )

        # calculate the log perplexity and then likelihood
        log_perplexity = model.log_perplexity(corpus_train)
        log_perp_values[i] = log_perplexity
    return log_perp_values

num_topics_range = np.arange(5, 41, 5).tolist()
log_perp_HighTech = find_perp(corpus_HighTech,num_topics_range,123)
log_perp_NonHighT = find_perp(corpus_NonHighT,num_topics_range,123)

import matplotlib.pyplot as plt

plt.plot(num_topics_range, log_perp_HighTech)
plt.xlabel("Num Topics")
plt.ylabel("Log Likelihood")
plt.legend(("Log Likelihood Values"), loc='best')
plt.show()

plt.plot(num_topics_range, log_perp_NonHighT)
plt.xlabel("Num Topics")
plt.ylabel("Log Likelihood")
plt.legend(("Log Likelihood Values"), loc='best')
plt.show()

# ---------------
# Fitting LDA 
# ---------------

model_HighTech = gensim.models.ldamodel.LdaModel(
                     corpus=corpus_HighTech, id2word=id2word, 
                     num_topics=13, random_state=123
                 )
model_HighTech.print_topics(10)

model_NonHighT = gensim.models.ldamodel.LdaModel(
                     corpus=corpus_NonHighT, id2word=id2word, 
                     num_topics=11, random_state=123
                 )
model_NonHighT.print_topics(10)
