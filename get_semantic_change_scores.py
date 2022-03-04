import argparse
import os
from semantic_change import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=str, default="sum")
    parser.add_argument("--method", type=str, default="apd")
    parser.add_argument("--reduction", type=str, default="pca")
    parser.add_argument("--silhouette", type=bool, default=True)
    parser.add_argument("--sem-eval", type=bool, default=False)
    parser.add_argument("--type", type=str, default="slang")
    parser.add_argument("--words", type=str, default="all_words.csv")
    args = parser.parse_args()

    # directory for results output
    if not os.path.exists("results"):
        os.mkdir("results")
    print("--- Getting data ---")
    if args.sem_eval:
        target_words, old_reps, new_reps = get_data_for_semeval(reps=args.reps, reps_abs_path="data/representations")
    else:
        old_reps, new_reps = get_data_for_tweets(type=args.type, path="data/")
        words_path = args.words
        selected_words_df = pd.read_csv(words_path)
        target_words = list(selected_words_df[selected_words_df.type == args.type].word)
    print("--- Scoring ---")
    if args.method == "apd":
        results = get_APD_scores(old_reps, new_reps, target_words)
        textfile = open("results/APD_results.txt", "w")

    elif args.method == "clustering":
        results = get_cluster_semantic_change_scores(old_reps, new_reps, target_words,
                                                         method=args.reduction,
                                                         silhouette=args.silhouette)
        if args.silhouette:
            textfile = open("results/KMeans_GMM_scores_" + args.reduction + "_silhouette_results.txt", "w")
        else:
            textfile = open("results/KMeans_GMM_scores_" + args.reduction + "_results.txt", "w")

    for elem in results:
        textfile.write(str(elem) + "\n")
    textfile.close()

