import pickle5 as pickle
from sklearn.decomposition import PCA
# internal imports
import config
from visualizations import plot_PCA_tradeoff

if __name__ == '__main__':

    old_reps_path = "../data/representations/old_slang_reps.pickle"
    new_reps_path = "../data/representations/old_slang_reps.pickle"

    with open(old_reps_path, "rb") as f:
        old_reps = pickle.load(f)

    with open(new_reps_path, "rb") as f:
        new_reps = pickle.load(f)

    plot_PCA_tradeoff(target="bromance", pca_model=PCA(), corpus1_reps=old_reps, corpus2_reps=new_reps)