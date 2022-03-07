import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from config import SLANG_COLOR, NONSLANG_COLOR, HYBRID_COLOR

#-------------------- plots of UD (Urban Dictionary) --------------------#
def plot_year_histogram(years, bin_num=10, color='mediumslateblue',
                        title="Distribution of years in which the definitions were posted"):
    fig, ax = plt.subplots(figsize=(4,1.7))

    ax.hist(years, bins=bin_num, ec="k", color=color)
    ax.locator_params(axis='y', integer=True)
    plt.title(title)
    plt.show()

def plot_numlikes_dislikes(defs):
    plt.plot(sorted(list(defs.num_likes))[::-1], color='mediumslateblue', label='number of upvotes')
    plt.plot(sorted(list(defs.num_dislikes))[::-1], color='darkorange', label='number of downvotes')
    plt.legend()
    plt.title("number of upvotes and downvotes in the dataset, in log scale")
    plt.yscale('log')
    plt.ylabel("log number of upvotes/downvotes")
    plt.show()

def plot_like_dislike_ratio(defs):
    EPSILON = 1e-20
    defs["like_dislike_ratio"] = defs["num_likes"] / (defs["num_dislikes"] + EPSILON)
    ratios = sorted(defs["like_dislike_ratio"])
    ratios = [a for a in ratios if a < 1 / EPSILON]
    plt.plot(ratios[::-1], color='mediumslateblue', label='upvotes/downvotes')
    plt.legend()
    plt.title("upvote/downvote ratio in the dataset")
    plt.yscale('log')
    plt.ylabel("log of upvote/downvote ratio")
    plt.show()

#-------------------- plots of representations & clusters-------------------#
class Label2Color:
    def __init__(self):
        self.NOISE = - 1
        self.DEFAULT_COLOR = 'xkcd:dark teal'
        self.color_dict = {
            self.NOISE : 'grey',
            0 : 'darkorange',
            1 : 'mediumslateblue',
            2 : 'gold',
            3 : 'fuchsia',
            5 : 'cyan',
            6 : 'forestgreen',
            7 : 'peru',
            8 : 'skyblue',
            9 : 'xkcd:amber',
            10 : 'xkcd:candy pink',
            11 : 'xkcd:sunny yellow'
        }

    def num2color(self, num):
        if num in self.color_dict:
            return self.color_dict[num]
        else:
            return self.DEFAULT_COLOR

def plot_PCA_tradeoff(target, pca_model, corpus1_reps, corpus2_reps):
    X1 = [elem.detach().numpy() for elem in corpus1_reps[target]]
    X2 = [elem.detach().numpy() for elem in corpus2_reps[target]]
    X = X1 + X2
    pca_model.fit(X)
    y = np.cumsum(pca_model.explained_variance_ratio_)
    x = np.arange(1, y.shape[0]+1)
    plt.plot(x, y,color="xkcd:periwinkle")
    plt.title(target)
    plt.axvline(x=np.where(y >= 0.7)[0][0], color="mediumslateblue")
    plt.axvline(x=np.where(y >= 0.8)[0][0], color="mediumslateblue")
    plt.axvline(x=np.where(y >= 0.9)[0][0], color="mediumslateblue")
    plt.show()

def plot_one_2d_rep(X_old, X_new, word, dim_reduction_method="pca"):
    X_old_0, X_old_1 = X_old[:,0], X_old[:,1]
    X_new_0, X_new_1 = X_new[:,0], X_new[:,1]

    plt.scatter(X_old_0, X_old_1, color='dodgerblue', label="old data", alpha=0.2)
    plt.scatter(X_new_0, X_new_1, color='gold', label="new data", alpha=0.2)
    plt.legend()
    plt.title("word representations of " + word + " in 2d with " + str(dim_reduction_method))
    plt.show()

def plot_clusters(X, labels, word, dim_reduction_method="pca"):
    label2color = Label2Color()
    X0, X1 = X[:,0], X[:,1]
    colors = [label2color.num2color(clss) for clss in labels]
    plt.scatter(X0, X1, c=colors)
    plt.title("cluster labels for representations of " + word + " in 2d with " + str(dim_reduction_method))
    plt.show()

def plot_2d_representations(X_old_pca,
                            X_new_pca,
                            X_old_tsne,
                            X_new_tsne,
                            dim_reduction_methods=["pca", "tsne"]):

    X_old_pca_0, X_old_pca_1 = X_old_pca[:,0], X_old_pca[:,1]
    X_new_pca_0, X_new_pca_1 = X_new_pca[:,0], X_new_pca[:,1]

    X_old_tsne_0, X_old_tsne_1 = X_old_tsne[:,0], X_old_tsne[:,1]
    X_new_tsne_0, X_new_tsne_1 = X_new_tsne[:,0], X_new_tsne[:,1]

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(X_old_pca_0, X_old_pca_1, color='dodgerblue', label="old data", alpha=0.2)
    axs[0].scatter(X_new_pca_0, X_new_pca_1, color='gold', label="new data", alpha=0.2)
    axs[0].legend()
    axs[0].set_title(str(dim_reduction_methods[0]))

    axs[1].scatter(X_old_tsne_0, X_old_tsne_1, color='dodgerblue', label="old data", alpha=0.2)
    axs[1].scatter(X_new_tsne_0, X_new_tsne_1, color='gold', label="new data", alpha=0.2)
    axs[1].legend()
    axs[1].set_title(str(dim_reduction_methods[1]))

    plt.show()

def draw_SSE_plot(data, max_n_clusters=8,
                max_iter=300, tol=1e-04, init='k-means++',
                n_init=10, algorithm='auto'):
    from sklearn.cluster import KMeans
    inertia_values = []
    for i in range(1, max_n_clusters+1):
        km = KMeans(n_clusters=i, max_iter=max_iter, tol=tol, init=init,
                    n_init=n_init, random_state=1, algorithm=algorithm)
        km.fit_predict(data)
        inertia_values.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(1, max_n_clusters+1), inertia_values, color='red')
    plt.xlabel('No. of Clusters', fontsize=15)
    plt.ylabel('SSE / Inertia', fontsize=15)
    plt.title('SSE / Inertia vs No. Of Clusters', fontsize=15)
    plt.grid()
    plt.show()

def draw_silhouette_plot(data, k_min=2, max_n_clusters=8,
                max_iter=300, tol=1e-04, init='k-means++',
                n_init=10, algorithm='auto'):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    for i in range(k_min, max_n_clusters+1):
        km = KMeans(n_clusters=i, max_iter=max_iter, tol=tol, init=init,
                    n_init=n_init, random_state=1, algorithm=algorithm)
        labels = km.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(k_min, max_n_clusters+1), silhouette_scores, color='red')
    plt.xlabel('No. of Clusters', fontsize=15)
    plt.ylabel('SSE / Inertia', fontsize=15)
    plt.title('SSE / Inertia vs No. Of Clusters', fontsize=15)
    plt.grid()
    plt.show()

def draw_cluster_score_plots(data, max_n_clusters=8,
                max_iter=300, tol=1e-04, init='k-means++',
                n_init=10, algorithm='auto', k_min=2):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    inertia_values, silhouette_scores = [], []
    for k in range(k_min, max_n_clusters+1):
        km = KMeans(n_clusters=k, max_iter=max_iter, tol=tol, init=init,
                    n_init=n_init, random_state=1111, algorithm=algorithm)
        labels = km.fit_predict(data)
        inertia_values.append(km.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
    fig, ax = plt.subplots(1,2, figsize=(8, 6))
    ax[0].plot(range(k_min, max_n_clusters+1), inertia_values, color='red')
    ax[0].set_title("SSE / Inertia score")

    ax[1].plot(range(k_min, max_n_clusters+1), silhouette_scores, color='blue')
    ax[1].set_title("Silhouette score")

    ax[0].set_xlabel('No. of Clusters', fontsize=15)
    ax[0].set_ylabel('Score', fontsize=15)

    ax[1].set_xlabel('No. of Clusters', fontsize=15)

    plt.suptitle('Score vs No. Of Clusters', fontsize=15)
    plt.grid()
    plt.show()

def plot_num_representations(new_nonslang_rep_num,
                             old_nonslang_rep_num,
                             new_slang_rep_num,
                             old_slang_rep_num,
                             title="number of representations \n for words with at least 150 representations",
                             ):
    plt.plot([k for k in range(len(new_nonslang_rep_num))],sorted(new_nonslang_rep_num), label="new nonslang")
    plt.plot([k for k in range(len(old_nonslang_rep_num))],sorted(old_nonslang_rep_num), label="old nonslang")
    plt.plot([k for k in range(len(new_slang_rep_num))],sorted(new_slang_rep_num), label="new slang")
    plt.plot([k for k in range(len(old_slang_rep_num))],sorted(old_slang_rep_num), label="old slang")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_old_vs_new(old_reps, new_reps, word):
    from utils import apply_PCA
    X1 = [elem.detach().numpy() for elem in old_reps[word]]
    X2 = [elem.detach().numpy() for elem in new_reps[word]]
    # X = X1 + X2
    X1 = apply_PCA(X1, dim=2)
    X2 = apply_PCA(X2, dim=2)
    # labels, K = get_labels(X)
    X_old = np.array(X1)
    X_new = np.array(X2)
    X_old_0, X_old_1 = X_old[:, 0], X_old[:, 1]
    X_new_0, X_new_1 = X_new[:, 0], X_new[:, 1]

    sns.scatterplot(X_old_0, X_old_1, color='mediumslateblue', label="2010 representations", alpha=0.8)
    sns.scatterplot(X_new_0, X_new_1, color='darkorange', label="2020 representations", alpha=0.8)
    plt.legend()
    plt.title("word representations of " + word + " in 2d with PCA")
    plt.show()

#-------------------- plots for statistical comparison: slang/nonslang  ------------------#

def plot_log_freqs(slang_freqs, nonslang_freqs, add_to_title=" in 2010"):
    plt.hist([[np.log(k) for k in slang_freqs],
              [np.log(k) for k in nonslang_freqs]],
             color=[SLANG_COLOR, NONSLANG_COLOR],
             label=["slang","nonslang"])
    plt.xlabel("log of # occurrences in 24 hours")
    plt.legend()
    plt.title("log-frequency of words"+add_to_title)
    plt.show()

def plot_polysemy(slang_polysemy_df,nonslang_polysemy_df,hybrid_polysemy_df=None):
    if hybrid_polysemy_df is not None:
        plt.hist([slang_polysemy_df.polysemy,nonslang_polysemy_df.polysemy, hybrid_polysemy_df.polysemy],
                 color=[SLANG_COLOR, NONSLANG_COLOR, HYBRID_COLOR],
                 label=["slang", "nonslang", "hybrid"])
        plt.legend()
        plt.title("Number of Word Senses - Distribution")
        plt.xlabel("# word senses")
        plt.show()
    else:
        plt.hist([slang_polysemy_df.polysemy, nonslang_polysemy_df.polysemy],
                 color=[SLANG_COLOR, NONSLANG_COLOR],
                 label=["slang", "nonslang"])
        plt.legend()
        plt.title("Number of Word Senses - Distribution")
        plt.xlabel("# word senses")
        plt.show()

def plot_3category_comparison(s_all_df, ns_all_df, h_all_df,
                              col="log_diff",
                              xlabel="log(2020 frequency/2010 frequency)",
                              title="Frequency change between 2010 and 2020"):
    plt.hist([s_all_df[col], ns_all_df[col], h_all_df[col]],
             label=["slang", "nonslang", "hybrid"],
             color=[SLANG_COLOR, NONSLANG_COLOR, HYBRID_COLOR],
             bins=22)
    plt.legend()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_slang_nonslang_comparison(s_all_df, ns_all_df, curr_col="log_diff",
                                   title="Frequency change between 2010 and 2020",
                                   xlabel="log(2020 frequency/2010 frequency)",
                                   bins=22):
    plt.hist([s_all_df[curr_col], ns_all_df[curr_col]],
             label=["slang", "nonslang"],
             color=[SLANG_COLOR, NONSLANG_COLOR],
             bins=bins)
    plt.legend()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_yearly_freqs(freq_df, words, colors, title_addition=" of words"):

    for word,color in zip(words,colors):
        freq_norm = freq_df.freq_norm[freq_df.word == word]
        if freq_norm > 0:
            plt.plot(freq_df.year[freq_df.word == word], freq_norm / max(freq_norm),
                     label=word,
                     color=color)

    plt.legend()
    plt.title("Yearly relative frequency" + title_addition)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    from collections import Counter
    from utils import normalize_values
    polysemy_WN = pd.read_csv("data/polysemy/polysemy_nonslang.csv")
    polysemy_MW = pd.read_csv("data/polysemy/polysemy_nonslang_MW.csv")
    polysemy_MW.columns = ["idx", "word", "polysemy_MW"]
    polysemy = pd.merge(polysemy_WN, polysemy_MW,on="word")
    polysemy["poly_diff"] = polysemy["polysemy_MW"] - polysemy["polysemy"]
    Counter(polysemy["poly_diff"])


    import pandas as pd
    data = pd.read_csv("data/causal_data_input_pos4_binary.csv")
    data["normalized_semantic_change"] = normalize_values(data["semantic_change"])
    data["log_diff"] = np.log(data["freq2020"]/data["freq2010"])
    data_slang = data[data["type"] == "slang"]
    data_nonslang = data[data["type"] == "nonslang"]
    plot_slang_nonslang_comparison(data_slang, data_nonslang, "normalized_semantic_change",
                                   title="Semantic change between 2010 and 2020",
                                   xlabel="Normalized semantic change score")
