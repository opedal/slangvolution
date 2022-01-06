def get_UD_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/all_definitions.dat")
    parser.add_argument("--name", type=str, default="UD_filtered_100000_sampled")
    parser.add_argument("--min-likes", type=int, default=20)
    parser.add_argument("--min-ratio", type=int, default=2)
    parser.add_argument("--sample", type=int, default=100000)
    args = parser.parse_args()
    fpath = "/Users/alacrity/Documents/GitHub/slangvolution/data/all_definitions.dat"
    defs = defs_to_pandas(fpath)

    filter_sample_and_write(defs, args.min_likes, args.min_ratio, name=args.name, sample_size=args.sample)

    PATH = "data/semeval2020_ulscd_eng/targets.txt"
    #target_words, old_reps, new_reps = get_data_for_semeval()
    target_words, old_reps, new_reps = get_data_for_tweets(type='slang')
    target_nonslang_words, old_nonslang_reps, new_nonslang_reps = get_data_for_tweets(type='nonslang')

    # res_old = inner_APD_scores(old_reps, target_words)
    # res_old_df = pd.DataFrame(res_old)
    # res_old_df.to_csv("semeval_APD_only2010.csv")
    #
    # res_new = inner_APD_scores(old_reps, target_words)
    # res_new_df = pd.DataFrame(res_new)
    # res_new_df.to_csv("semeval_APD_only2020.csv")

    res = get_APD_scores(old_reps, new_reps, target_words, min_tweets=config.MIN_TWEETS_PER_WORD)
    res_between_df = pd.DataFrame(res)
    res_between_df.to_csv("slang_APD.csv")

    # results = get_cluster_semantic_change_scores(old_reps, new_reps, target_words, "pca", normalize=True)
    # print("Targets scored: ", len(results))
    # textfile = open("APD_more_results.txt", "w")
    # for elem in results:
    #     textfile.write(str(elem) + "\n")
    # textfile.close()