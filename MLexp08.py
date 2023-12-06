import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Define the placeholders for the input and output data
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(0.5, name="weights")
model = tf.add(tf.multiply(x, w), 0.5)
cost = tf.reduce_mean(tf.square(model - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Toy dataset
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]
# Create a TensorFlow session
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    # Training loop
    for i in range(1000):
        sess.run(train, feed_dict={x: x_train, y: y_train})
    # Evaluate the model
    w_val = sess.run(w)
    # Plot the loss surface
    w_values = np.linspace(-1, 2, 100)
    loss_values = []
    for w_val in w_values:
        loss_val = sess.run(cost, feed_dict={x: x_train, y: y_train, w: w_val})
        loss_values.append(loss_val)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(w_values, loss_values, label='Loss Surface')
    plt.xlabel('Weight (w)')
    plt.ylabel('Loss')
    plt.title('Loss Surface')
    plt.legend()
    # Plot the trajectory of gradient descent
    plt.subplot(1, 2, 2)
    plt.plot(w_values, loss_values, label='Loss Surface')
    plt.scatter(w_val, sess.run(cost, feed_dict={x: x_train, y: y_train}), color='red', label='Final Weight')
    plt.xlabel('Weight (w)')
    plt.ylabel('Loss')
    plt.title('Trajectory of Gradient Descent')
    plt.legend()
    plt.show()
    import numpy as np
import matplotlib.pyplot as plt
# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = 4 + 3 * X + np.random.randn(100, 1)
# Initialize model parameters
w = np.random.randn(2, 1)
b = np.random.randn(1)[0]
# Set the learning rate
alpha = 0.1
# Set the number of iterations
num_iterations = 20
# Create a mesh to plot the loss surface
w1, w2 = np.meshgrid(np.linspace(-5, 5, 100),
np.linspace(-5, 5, 100))
# Compute the loss for each point on the grid
loss = np.zeros_like(w1)
for i in range(w1.shape[0]):
for j in range(w1.shape[1]):
loss[i, j] = np.mean((y - w1[i, j] * X - w2[i, j] * X**2)**2)
# Perform gradient descent
for i in range(num_iterations):
# Compute the gradient of the loss
# with respect to the model parameters
grad_w1 = -2 * np.mean(X * (y - w[0]* X - w[1] * X**2))
grad_w2 = -2 * np.mean(X**2 * (y - w[0]* X - w[1] * X**2))
# Update the model parameters
w[0] -= alpha * grad_w1
w[1] -= alpha * grad_w2
# Plot the loss surface
fig = plt.figure(figsize=(30, 20))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(w1, w2, loss, cmap='coolwarm')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.plot(w[0], w[1], np.mean((y - w[0] * X - w[1] * X**2)**2),'o', c='red', markersize=50)
plt.show()
