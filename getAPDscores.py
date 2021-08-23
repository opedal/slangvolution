import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.evaluate import permutation_test
## inner imports
from semantic_change_scores import get_data_for_tweets, get_APD_scores, MIN_TWEETS

target_words, old_reps, new_reps = get_data_for_tweets(type='slang')
target_nonslang_words, old_nonslang_reps, new_nonslang_reps = get_data_for_tweets(type='nonslang')

res = get_APD_scores(old_reps, new_reps, target_words, min_tweets=MIN_TWEETS)
res_between_df = pd.DataFrame(res)
res_between_df.to_csv("slang_APD_scores.csv")
#res_between_df = pd.read_csv("slang_APD_scores.csv")

ns_res = get_APD_scores(old_nonslang_reps, new_nonslang_reps, target_nonslang_words, min_tweets=MIN_TWEETS)
ns_res_between_df = pd.DataFrame(ns_res)
ns_res_between_df.to_csv("nonslang_APD_scores.csv")
#ns_res_between_df = pd.read_csv("nonslang_APD_scores.csv")

slang_APD = res_between_df.combined_APD
nonslang_APD = ns_res_between_df.combined_APD


p_value_mlxtend = permutation_test(slang_APD, nonslang_APD,
                           method='approximate',
                           num_rounds=10000,
                           seed=0)
print("p val from mlxtend=",p_value_mlxtend)
sns.histplot(res_between_df.combined_APD, color='darkorange', label='slang')
sns.histplot(ns_res_between_df.combined_APD, color='mediumslateblue', label='nonslang')
plt.legend()
plt.title("Combined APD between 2010-2020 \n for slang versus nonslang words")
plt.show()

plt.hist([ns_res_between_df.combined_APD,res_between_df.combined_APD], bins=22, color=['darkorange','mediumslateblue'],label=['nonslang', 'slang'])
plt.legend()
plt.title("Combined APD between 2010-2020 \n for slang versus nonslang words")
plt.show()