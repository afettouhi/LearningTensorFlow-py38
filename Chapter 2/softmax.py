import tensorflow as tf
import tensorflow_datasets as tfds

DATA_DIR = '../data/'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = tfds.load(name='mnist', split=['train', 'test'], data_dir=DATA_DIR)

tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))

gd_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.compat.v1.Session() as sess:
    # Train
    sess.run(tf.compat.v1.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = tfds.load(name='mnist', split=['train', 'test'],
                                       data_dir=DATA_DIR, batch_size=MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans * 100))
