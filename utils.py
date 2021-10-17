from sklearn.decomposition import PCA
import os
import pandas as pd
from scipy.stats import ttest_ind
from mlxtend.evaluate import permutation_test

def apply_PCA(data, dim=50):
    pca_model = PCA(n_components=dim)
    return pca_model.fit_transform(data)

def apply_UMAP(data, dim = 50, n_neighbors=15, min_dist = 0.1):
    import umap.umap_ as umap
    umap_model = umap.UMAP(n_components=dim, n_neighbors=n_neighbors, min_dist=min_dist)
    return umap_model.fit_transform(data)

def check_num_tweets_collected(data_path, words_list):
    print("there are", len(os.listdir(data_path)), "files in", data_path)
    k = 0
    for word in words_list:
        word_df_path = os.path.join(data_path, "tweets_df_" + str(word) + ".csv")
        try:
            word_df = pd.read_csv(word_df_path)
        except:
            print(word, "doesn't exist yet")
            continue
        if len(word_df.word) < 200:
            print(word, "only has", len(word_df.word), "tweets")
        else:
            k +=1
    print("there are", k, "words with more than 200 tweets collected")

def independence_tests(slang_scores, nonslang_scores):
    t_test_statistic, t_test_pval = ttest_ind(slang_scores,nonslang_scores)
    perm_test_pval = permutation_test(slang_scores, nonslang_scores,
                                      method="approximate", seed=111, num_rounds=10000
                                      )
    print("t-test p-value is", t_test_pval, "and permutation test p-value is", perm_test_pval)