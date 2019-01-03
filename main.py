import argparse
import sys
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
FLAGS = None

def model(images):
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps" : ps_hosts, "worker" : worker_hosts})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name =="worker":

        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

            # loss = ...
            mnist = read_data_sets("./dataset", one_hot=True)
            images = tf.placeholder(tf.float32, [None, 784]) # n * 784?
            labels = tf.placeholder(tf.int32, [None, 10]) # n * 10?

            logits = model(images)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
            hooks = [tf.train.StopAtStepHook(last_step=2000)]
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)
            train_op = optimizer.minimize(loss, global_step=global_step, aggregation_method=tf.AggregationMethod.ADD_N)
            
            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), checkpoint_dir="./checkpoint_dir", hooks=hooks) as mon_sess:

                while not mon_sess.should_stop():
                    img_batch, label_batch = mnist.train.next_batch(32)
                    _, ls, step = mon_sess.run([train_op, loss, global_step],
                                                feed_dict={images: img_batch, labels: label_batch})
                    
                    if step % 100 == 0:
                        print("Train step %d, loss: %f" % (step, ls))
                                                
            print("say hi")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == "true")

    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
  # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
