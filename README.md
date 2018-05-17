# Description
Solving a two classes non-linearly separable classification problem using a MLP. The problem consists of four semicircle centered at the points +1 and -1 at each axis. The random data generated that falls inside any intersection of the circles belongs to a class and the remaining data to another class.

# How to use
run gen_data.py with python 3.5 and it will genrate a file with the random data: the training data, test data and labels. Then, run train_MLP.py to train the MLP network, it can take some time, and will plot the results. The results will be saved in a file to be plotted in the future by running plot_result.py.
