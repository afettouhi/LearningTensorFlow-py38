import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.disable_eager_execution()

BATCH_SIZE = 50
TRAINING_STEPS = 5000
PRINT_EVERY = 100
LOG_DIR = "../data/log"

parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
           "localhost:2224",
           "localhost:2225"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

tf.compat.v1.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.compat.v1.flags.DEFINE_integer("task_index", 0, "Index of task")
FLAGS = tf.compat.v1.flags.FLAGS

server = tf.compat.v1.train.Server(cluster,
                                   job_name=FLAGS.job_name,
                                   task_index=FLAGS.task_index)

mnist = tfds.load(name='mnist', data_dir='../data/mnist')


def net(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    net = tf.keras.layers.Conv2D(x_image, 32, [5, 5], scope='conv1')
    net = tf.keras.layers.MaxPooling2D(net, [2, 2], scope='pool1')
    net = tf.keras.layers.Conv2D(net, 64, [5, 5], scope='conv2')
    net = tf.keras.layers.MaxPooling2D(net, [2, 2], scope='pool2')
    net = tf.keras.layers.Flatten(net, scope='flatten')
    net = tf.keras.layers.Dense(net, 500, scope='fully_connected')
    net = tf.keras.layers.Dense(net, 10, activation=None, scope='pred')
    return net


if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":

    with tf.device(tf.compat.v1.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        global_step = tf.compat.v1.get_variable('global_step', [],
                                                initializer=tf.constant_initializer(0),
                                                trainable=False)

        x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="x-input")
        y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name="y-input")
        y = net(x)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                                               labels=y_))

        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                                     global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.compat.v1.global_variables_initializer()

    sv = tf.compat.v1.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                       logdir=LOG_DIR,
                                       global_step=global_step,
                                       init_op=init_op)

    with sv.managed_session(server.target) as sess:
        step = 0

        while not sv.should_stop() and step <= TRAINING_STEPS:

            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

            _, acc, step = sess.run([train_step, accuracy, global_step],
                                    feed_dict={x: batch_x, y_: batch_y})

            if step % PRINT_EVERY == 0:
                print("Worker : {}, Step: {}, Accuracy (batch): {}".
                      format(FLAGS.task_index, step, acc))

        test_acc = sess.run(accuracy,
                            feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Test-Accuracy: {}".format(test_acc))

    sv.stop()
