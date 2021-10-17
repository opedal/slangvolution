from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from mlxtend.evaluate import permutation_test
from visualizations import plot_log_freqs_change, plot_polysemy

def independence_tests(slang_scores, nonslang_scores):
    t_test_statistic, t_test_pval = ttest_ind(slang_scores,nonslang_scores)
    perm_test_pval = permutation_test(slang_scores, nonslang_scores,
                                      method="approximate", seed=111, num_rounds=10000
                                      )
    print("t-test p-value is", t_test_pval, "and permutation test p-value is", perm_test_pval)

if __name__ == '__main__':

    polysemy_file_paths = {"slang": "word-lists/polysemy_slang.csv",
                           "nonslang": "word-lists/polysemy_nonslang.csv",
                           "hybrid": "word-lists/polysemy_hybrid.csv",
                           }

    freq_file_paths = {"slang": "data/frequencies/freq_slang_counts_24h_2010.csv",
                       "nonslang": "data/frequencies/freq_nonslang_counts_24h_2010.csv"
                       }

    causal_df = pd.read_csv("word-lists/causal_data_input.csv")

    hybrid_polysemy_df = pd.read_csv(polysemy_file_paths["hybrid"])
    hybrid_polysemy_df["polysemy"] = hybrid_polysemy_df["num_s"] +  hybrid_polysemy_df["num_ns"]
    hybrid_polysemy_df.to_csv(polysemy_file_paths["hybrid"])

    slang_freq_df = pd.read_csv(freq_file_paths["slang"])
    nonslang_freq_df = pd.read_csv(freq_file_paths["nonslang"])

    plot_log_freqs_change(slang_freq_df=slang_freq_df, nonslang_freq_df=nonslang_freq_df)

    slang_polysemy_df = pd.read_csv(polysemy_file_paths["slang"])
    nonslang_polysemy_df = pd.read_csv(polysemy_file_paths["nonslang"])

    independence_tests(slang_polysemy_df.polysemy.values, nonslang_polysemy_df.polysemy.values)
    plot_polysemy(slang_polysemy_df,nonslang_polysemy_df,hybrid_polysemy_df)
