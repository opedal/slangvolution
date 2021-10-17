import pandas as pd
from visualizations import plot_log_freqs, plot_polysemy
from utils import independence_tests

if __name__ == '__main__':

    polysemy_file_paths = {"slang": "data/polysemy/polysemy_slang.csv",
                           "nonslang": "data/polysemy/polysemy_nonslang.csv",
                           "hybrid": "data/polysemy/polysemy_hybrid.csv",
                           }

    freq_file_paths = {"slang 2010": "data/frequencies/freq_slang_counts_24h_2010.csv",
                       "slang 2020": "data/frequencies/freq_slang_counts_24h_2020.csv",
                       "nonslang 2010": "data/frequencies/freq_nonslang_counts_24h_2010.csv",
                       "nonslang 2020": "data/frequencies/freq_nonslang_counts_24h_2020.csv",
                       }

    causal_df = pd.read_csv("../data/causal_data_input.csv")

    hybrid_polysemy_df = pd.read_csv(polysemy_file_paths["hybrid"])
    hybrid_polysemy_df["polysemy"] = hybrid_polysemy_df["num_s"] +  hybrid_polysemy_df["num_ns"]
    hybrid_polysemy_df.to_csv(polysemy_file_paths["hybrid"])

    slang_freq_df = pd.read_csv(freq_file_paths["slang 2010"])
    nonslang_freq_df = pd.read_csv(freq_file_paths["nonslang 2010"])

    print("Comparing 2010 Frequencies:")
    plot_log_freqs(slang_freqs=slang_freq_df.freq.values, nonslang_freqs=nonslang_freq_df.freq.values)
    independence_tests(slang_freq_df.freq.values, nonslang_freq_df.freq.values)

    slang_polysemy_df = pd.read_csv(polysemy_file_paths["slang"])
    nonslang_polysemy_df = pd.read_csv(polysemy_file_paths["nonslang"])

    print("Comparing Polysemy:")
    independence_tests(slang_polysemy_df.polysemy.values, nonslang_polysemy_df.polysemy.values)
    plot_polysemy(slang_polysemy_df,nonslang_polysemy_df,hybrid_polysemy_df)
