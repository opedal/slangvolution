from sklearn.decomposition import PCA
import os
import pandas as pd

def apply_PCA(data, dim=50):
    pca_model = PCA(n_components=dim)
    return pca_model.fit_transform(data)

def apply_UMAP(data, dim = 50, n_neighbors=15, min_dist = 0.1):
    import umap.umap_ as umap
    umap_model = umap.UMAP(n_components=dim, n_neighbors=n_neighbors, min_dist=min_dist)
    return umap_model.fit_transform(data)

def check_folder(data_path, slang=True):
    print("there are", len(os.listdir(data_path)), "files in", data_path)
    selected_words_df = pd.read_csv("selected_words.csv")
    if slang:
        words_list = list(selected_words_df.slang)
    else:
        words_list = list(selected_words_df.nonslang)

    for word in words_list:
        word_df_path = os.path.join(data_path, "tweets_df_" + str(word) + ".csv")
        try:
            word_df = pd.read_csv(word_df_path)
        except:
            print(word, "is not working")
            continue
        if len(word_df.word) < 200:
            print(word, "only has", len(word_df.word), "tweets")
