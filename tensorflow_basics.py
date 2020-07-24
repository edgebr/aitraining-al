# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiments
#
# This experiment guide includes nine experiments, introducing basic equipment operation and
# Configuration, TensorFlow's helloworld, sessions, matrix multiplication, TensorFlow
# Virtualization, and housing price prediction.
#
# - Experiment 1: "Hello, TensorFlow".
# - Experiment 2: Understand functions of sessions through a session experiment using the with session function.
# - Experiment 3: Understand matrix multiplication by multiplying two matrices with ranks of tensors greater than 2.
# - Experiment 4: Understand the definition of variables. 
#   - Define variables with Variable and get_variable respectively and observe the difference between these two methods.
# - Experiment 5: Understand the visualization of TensorBoard. 
#   - TensorBoard aggregates all kinds of data into a log file. 
#   - You can enable TensorBoard service to read the log file and enable the 6060 port to provide web services so that users can view data via a browser.
# - Experiment 6: Understand data reading and processing by reading .csv files and displaying them based on given conditions.
# - Experiment 7: Understand graphic operations. 
#   - Create a graph in three ways and set it as the default graph. Use the get_default_graph() function to access the default graph and verify its settings.
# - Experiment 8: Understand save and use of models. 
#   - After importing data, analyze data characteristics and define variables based on the characteristics. 
#   - Create a model and define output nodes. 
#   - Build the structure for forward propagation and then the structure for backpropagation. 
#   - Compile and train the model to get appropriate parameters. 
#   - After training data and testing the model, create a saver and a path to save parameters in the session automatically. 
#   - When the model is saved, you can access the model for use.
# - Experiment 9: A comprehensive experiment of forecasting housing price through the instantiation of linear regression. 
#   - Use the dataset of housing prices in Beijing and skills in the prior eight experiments to forecast the housing price.

# %% [markdown]
# ## Experiment 1
# %%
import tensorflow as tf

# %%
# Defining a variable
hello = tf.constant('hello, tensorflow!')  # a constant 


# %%
sess = tf.Session()     # Creates a session
print(sess.run(hello))  # Run the session on the `hello` constant to get the result
# %%
sess.close()  # Close the session
# %% [markdown]
# ## Experiment 2
# - After this experiment you will understand the definition of sessions and how to use them with the python context manager (`with`).

# %% 
import tensorflow as tf

# %%
# Defining constants
a = tf.constant(3)
b = tf.constant(4)

# %%
# Creating a Session
with tf.Session() as sess:  # `with` starts a context where Session will be automatically closed
    print(f'Add: {sess.run(a + b)}')
    print(f'Multiply: {sess.run(a * b)}')


# %% [markdown]
# ## Experiment 3
# - After this experiment you will understand the "tensor" part of TensorFlow and how to use TensorFlow to multiply matrices.

# %%
import tensorflow as tf

# %%
# Start a TF default session
sess = tf.InteractiveSession()

# %%
# Creates two matrix variables
w1 = tf.Variable(tf.random_normal(shape=[2, 3], mean=1.0, stddev=1.0))
w2 = tf.Variable(tf.random_normal(shape=[3, 1], mean=1.0, stddev=1.0))

# %%
# Defining a constant matrix
x = tf.constant([[0.7, 0.9]])

# %%
# Initializing global variables: w1, w2
tf.global_variables_initializer().run()

# %%
# Multiply matrices
a = tf.matmul(x, w1)
b = tf.matmul(a, w2)
print(b.eval())  # Evaluates tensor `b` in the session

# %% [markdown]
# ## Experiment 4

# %% [markdown]
# ## Experiment 5

# %% [markdown]
# ## Experiment 6

# %% [markdown]
# ## Experiment 7

# %% [markdown]
# ## Experiment 8

# %% [markdown]
# ## Experiment 9

