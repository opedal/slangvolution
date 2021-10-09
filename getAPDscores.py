import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.evaluate import permutation_test
## inner imports
from semantic_change_scores import get_data_for_tweets, get_APD_scores, MIN_TWEETS

#old_slang_reps, new_slang_reps = get_data_for_tweets(type='slang')
#old_nonslang_reps, new_nonslang_reps = get_data_for_tweets(type='nonslang')

#words_path = "word-lists/all_words_300.csv"
#all_words_df = pd.read_csv(words_path)
#slang_word_list = list(all_words_df[all_words_df.type == "slang"].word)
#nonslang_word_list = list(all_words_df[all_words_df.type == "nonslang"].word)

#res = get_APD_scores(old_slang_reps, new_slang_reps, slang_word_list, min_tweets=MIN_TWEETS)
#res_slang = pd.DataFrame(res)

#res = get_APD_scores(old_nonslang_reps, new_nonslang_reps, nonslang_word_list, min_tweets=MIN_TWEETS)
#res_nonslang = pd.DataFrame(res)

#res_all = pd.concat([res_slang, res_nonslang])
#all_words_df = all_words_df.merge(res_all, left_on="word", right_on="word", how="left")
#all_words_df.to_csv("word-lists/all_words_300_change_scores.csv", index=False)

#slang_APD = res_slang.combined_APD
#nonslang_APD = res_nonslang.combined_APD

all_words_df = pd.read_csv("word-lists/all_words_300_change_scores.csv")
slang_APD = list(all_words_df[all_words_df.type == "slang"].combined_APD)
nonslang_APD = list(all_words_df[all_words_df.type == "nonslang"].combined_APD)

p_value_mlxtend = permutation_test(slang_APD, nonslang_APD,
                           method='approximate',
                           num_rounds=10000,
                           seed=0)
print("p val from mlxtend=",p_value_mlxtend)
sns.histplot(slang_APD, color='darkorange', label='slang')
sns.histplot(nonslang_APD, color='mediumslateblue', label='nonslang')
plt.legend()
plt.title("Combined APD between 2010-2020 \n for slang versus nonslang words")
plt.show()

plt.hist([nonslang_APD,slang_APD], bins=22, color=['darkorange','mediumslateblue'],label=['nonslang', 'slang'])
plt.legend()
plt.title("Combined APD between 2010-2020 \n for slang versus nonslang words")
plt.show()