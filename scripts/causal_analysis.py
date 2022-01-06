import pandas as pd
from os import path as osp
# Internal Imports
import config
from utils import independence_tests
from visualizations import plot_log_freqs, plot_polysemy

if __name__ == '__main__':
    polysemy_folder_path = "../data/polysemy"
    freqs_folder_path = "../data/frequencies"
    freq_file_names = config.FREQ_FILE_NAMES
    polysemy_file_names = config.POLYSEMY_FILE_NAMES

    causal_df = pd.read_csv("../data/causal_data_input.csv")

    slang_freq_df = pd.read_csv(osp.join(freqs_folder_path, freq_file_names["slang2010"]))
    nonslang_freq_df = pd.read_csv(osp.join(freqs_folder_path, freq_file_names["nonslang2010"]))

    print("Comparing 2010 Frequencies:")
    plot_log_freqs(slang_freqs=slang_freq_df.freq.values, nonslang_freqs=nonslang_freq_df.freq.values)
    independence_tests(slang_freq_df.freq.values, nonslang_freq_df.freq.values)

    hybrid_polysemy_df = pd.read_csv(osp.join(polysemy_folder_path, polysemy_file_names["hybrid"]))
    slang_polysemy_df = pd.read_csv(osp.join(polysemy_folder_path, polysemy_file_names["slang"]))
    nonslang_polysemy_df = pd.read_csv(osp.join(polysemy_folder_path, polysemy_file_names["nonslang"]))

    print("Comparing Polysemy:")
    independence_tests(slang_polysemy_df.polysemy.values, nonslang_polysemy_df.polysemy.values)
    plot_polysemy(slang_polysemy_df,nonslang_polysemy_df,hybrid_polysemy_df)
