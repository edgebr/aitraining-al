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
hello = tf.constant("hello, tensorflow!")  # a constant


# %%
sess = tf.Session()  # Creates a session
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
    print(f"Add: {sess.run(a + b)}")
    print(f"Multiply: {sess.run(a * b)}")


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
# - After this experiment you will understand `tf.Variable` and the `get_variable` function.

# %%
import tensorflow as tf

# %%
# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# %%
var1 = tf.Variable(10.0, name="varname")
var2 = tf.Variable(11.0, name="varname")
var3 = tf.Variable(12.0)
var4 = tf.Variable(13.0)

# %%
# Variable scope allows you to create new variables and to share already created ones
# while providing checks to not create or share by accident.
# TODO: make it more clear
with tf.variable_scope("test1"):
    var5 = tf.get_variable("varname", shape=[2], dtype=tf.float32)

with tf.variable_scope("test2"):
    var6 = tf.get_variable("varname", shape=[2], dtype=tf.float32)


# %%
print("var1: ", var1.name)
print(
    "var2: ", var2.name
)  # A tf variable with a existing name gets a suffix to differentiate between them
print("var3: ", var3.name)
print("var4: ", var4.name)
print(
    "var5: ", var5.name
)  # With `variable_scope` we can enclose a variable within a desired scope
print("var6: ", var6.name)
# %% [markdown]
# ## Experiment 5
# - After this experiment you will understand the virtualization tool TensorBoard.

# %%
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# %%
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    ma = [val if idx < w else sum(a[(idx - w) : idx]) / w for idx, val in enumerate(a)]
    return ma


# %%
x_train = np.linspace(-1, 1, 100)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.3  # y = 2 * x + noise

# %%
plt.plot(x_train, y_train, "ro", label="Original data")
plt.legend()
plt.show()

tf.reset_default_graph()

# %%
# Creating a model
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Model parameters
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1], name="bias"))

# %%
z = tf.multiply(X, W) + b
tf.summary.histogram("z", z)

# %%
# Reverse optimization

# Cost function
cost = tf.reduce_mean(tf.square(Y - z))
tf.summary.scalar("loss_function", cost)

# Gradient descent
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# %%
# Start a session
init = tf.global_variables_initializer()
plot_data = {"batch_size": [], "loss": []}
with tf.Session() as sess:
    sess.run(init)
    # Merge all summaries
    merged_summary_op = tf.summary.merge_all()
    # Create summary writer for the writing
    summary_writer = tf.summary.FileWriter(f"log/run-{time.time_ns()}", sess.graph)

    # Write data to the model
    training_epochs = 15
    display_step = 1
    for epoch in range(training_epochs):
        for (x, y) in zip(x_train, y_train):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            summary_writer.add_summary(summary_str, epoch)

        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            weights = sess.run(W)
            bias = sess.run(b)
            print(f"Epoch: {epoch + 1} cost={loss}, W={weights}, b={bias}")
            if loss:
                plot_data["batch_size"].append(epoch)
                plot_data["loss"].append(loss)

    print("Finished!")
    cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    weights = sess.run(W)
    bias = sess.run(b)
    print(f"cost={cost}, W={weights}, b={bias}")


# %%
# Visualize results
y_pred = weights * x_train + bias
plot_data["avgloss"] = moving_average(plot_data["loss"])

plt.subplot(211)
plt.plot(x_train, y_train, "ro", markersize=4, label="Original data")
plt.plot(x_train, y_pred, label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.subplot(212)
plt.plot(plot_data["batch_size"], plot_data["avgloss"], "b--")
plt.xlabel("Minibatch number")
plt.ylabel("Loss")
plt.title("Minibatch run vs Training loss")

plt.show()


# %% [markdown]
# Now, in your terminal, type: `tensorboard --logdir log` and go to the given address. You can see your training log!

# %% [markdown]
# ## Experiment 6
# - After this experiment, you will understand how to read data from files with TensorFlow.
# - TODO: use `tf.data` and `tf.data.TextLineDataset`.

# %%
import tensorflow as tf


# %%
data = tf.train.string_input_producer(["data.csv"])
reader = tf.TextLineReader()

# Getting queue values
key, value = reader.read(data)
# key represents the information of the read file and the number of rows.
# value represents the raw strings read by row, which are sent to the decoder for decoding.

# The data type here determines the type of data to be read, which should be in the list form.
record_defaults = [[1.0], [1.0], [1.0], [1.0]]
# Each parsed attribute (column) is a scalar with the rank value of 0
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3])

# %%
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

# %%
with tf.Session() as sess:
    # Start a session and perform initialization
    sess.run(init_op)
    sess.run(local_init_op)
    # Start populating the filename queue
    coord = tf.train.Coordinator()
    # Feed the queue
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(30):
        example, label = sess.run([features, col4])
        print(example, label)

    print("Done!")

    coord.request_stop()
    coord.join(threads)

# %% [markdown]
# ## Experiment 7
# - After this exepriment, you will understand graphic operations with TensorFlow. That is, oprations within graphs.

# %%
import numpy as np
import tensorflow as tf

# %%
# Defines a constant variable
c = tf.constant(0.0)
# Creates a graph
g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    # Not the same graph as c1.graph and g
    print(c.graph)

# Same graph as c.graph
g2 = tf.get_default_graph()
print(g2)

# Reset graphs
tf.reset_default_graph()
g3 = tf.get_default_graph()
print(g3)  # New graph


# %%
# Get the tensor
print(c1.name)
t = g.get_tensor_by_name(name="Const:0")  # This name is the default one.
print(t)

# %%
# Get an operation

# Define constant variables
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

# Define a op named 'example_op'
tensor1 = tf.matmul(a, b, name="example_op")
# Print op.name and itself (and break line)
print(tensor1.name, tensor1)
# Get same op as above using its output tensor name
test = g3.get_tensor_by_name("example_op:0")
print(test)

print(tensor1.op.name)
test_op = g3.get_operation_by_name("example_op")
print(test_op)

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    test = tf.get_default_graph().get_tensor_by_name("example_op:0")
    print(test)

# TODO: improve this output

# %%
# Get all lists

# Return the list of operating nodes in the graph
tt2 = g.get_operations()
print(tt2)

# %%
# Get an object
tt3 = g.as_graph_element(c1)
print(tt3)

# %% [markdown]
# ## Experiment 8

# %% [markdown]
# ## Experiment 9
