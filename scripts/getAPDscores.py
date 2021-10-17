import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.evaluate import permutation_test
## internal imports
from semantic_change import get_data_for_tweets, get_APD_scores
from config import MIN_TWEETS_PER_WORD, SLANG_COLOR, NONSLANG_COLOR

### Get Representations
old_slang_reps, new_slang_reps = get_data_for_tweets(type='slang')
old_nonslang_reps, new_nonslang_reps = get_data_for_tweets(type='nonslang')
old_hybrid_reps, new_hybrid_reps = get_data_for_tweets(type='hybrid')

words_path = "../data/word-lists/all_words_300_change_scores.csv"
all_words_df = pd.read_csv(words_path)
slang_word_list = list(all_words_df[all_words_df.type == "slang"].word)
nonslang_word_list = list(all_words_df[all_words_df.type == "nonslang"].word)
hybrid_word_list = list(all_words_df[all_words_df.type == "both"].word)

res = get_APD_scores(old_hybrid_reps, new_hybrid_reps, hybrid_word_list, min_tweets=MIN_TWEETS_PER_WORD)
res_hybrid = pd.DataFrame(res)

res = get_APD_scores(old_nonslang_reps, new_nonslang_reps, nonslang_word_list, min_tweets=MIN_TWEETS_PER_WORD)
res_nonslang = pd.DataFrame(res)

res = get_APD_scores(old_slang_reps, new_slang_reps, slang_word_list, min_tweets=MIN_TWEETS_PER_WORD)
res_slang = pd.DataFrame(res)

#res_all = pd.concat([res_slang, res_nonslang])
all_words_df = all_words_df.merge(res_hybrid, on=["word",], how="left")
all_words_df.to_csv("word-lists/all_words_300_change_scores2.csv", index=False)

slang_APD = res_slang.combined_APD
nonslang_APD = res_nonslang.combined_APD

p_value_mlxtend = permutation_test(slang_APD, nonslang_APD,
                           method='approximate',
                           num_rounds=10000,
                           seed=0)

print("permutation test p-val=", p_value_mlxtend)
sns.histplot(slang_APD, color=SLANG_COLOR, label='slang')
sns.histplot(nonslang_APD, color=NONSLANG_COLOR, label='nonslang')
plt.legend()
plt.title("Combined APD between 2010-2020 \n for slang versus nonslang words")
plt.show()

plt.hist([nonslang_APD,slang_APD], bins=22, color=[SLANG_COLOR,NONSLANG_COLOR],label=['nonslang', 'slang'])
plt.legend()
plt.title("Combined APD between 2010-2020 \n for slang versus nonslang words")
plt.show()