import pandas as pd

# helper function to find related terms in a given word2vec model
def find_terms(wv, focal_terms, num_words, outfilename):
    data = pd.DataFrame(columns=['term1','term2','similarity','rank'])
    for w in focal_terms:

        # filter out words not in the library
        if not w in wv.key_to_index:
            print("Word '" + w + "' does not appear in this model.")
        else:
            sim_words = wv.most_similar(positive=w, topn=num_words)
            subdf = pd.DataFrame(sim_words, \
                                 columns =['term2', 'similarity'])
            subdf['term1'] = w 
            subdf['rank'] = range(1,num_words+1)
            subdf = subdf[['term1','term2','similarity','rank']]
            data = pd.concat([data, subdf], ignore_index=True, axis=0)
    data.to_csv(outfilename, index=False)

if __name__ == "__main__":

    # ---------------------------------------------------
    # Finding Related Words Using Pretrained Word Vectors
    # ---------------------------------------------------

    # Here we define our focal terms:
    focal_terms = ['innovation', 'innovations', 'innovate', 'innovates', 
                   'innovative', 'innovating'] # 'innovated', 'innovator'

    # download Google's pretrained model
    import gensim.downloader
    pretrained_wv = gensim.downloader.load('word2vec-google-news-300')

    # find related terms by Google's pre-trained word vectors
    find_terms(pretrained_wv, focal_terms, 100, "pretrained_wv_results.csv")

    # ---------------------------------------------------
    # Training Your Own Word2Vec Model
    # ---------------------------------------------------

    df = pd.read_csv("../data.csv", encoding_errors="ignore")
    docs = df['text'].tolist()

    import gensim
    my_text = [gensim.utils.simple_preprocess(x) for x in docs]

    # custom trained word vector with length 300
    from gensim.models import Word2Vec
    model_custom = Word2Vec(sentences=my_text, vector_size=300, 
                            window=5, min_count=1, workers=4)

    # find related terms by custom-trained word vectors
    find_terms(model_custom.wv, focal_terms, 100, "custrained_wv_results.csv")
