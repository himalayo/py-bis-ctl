import tensorflow as tf
import matplotlib.pyplot as plt

def parse_example(ex,cardinal=1200):
    out = tf.io.parse_example(ex,{
        'ref': tf.io.FixedLenFeature(shape=cardinal,dtype=tf.float32),
        'bis': tf.io.FixedLenFeature(shape=cardinal,dtype=tf.float32),
        'prop': tf.io.FixedLenFeature(shape=cardinal,dtype=tf.float32),
        'remi': tf.io.FixedLenFeature(shape=cardinal,dtype=tf.float32),
        'z': tf.io.FixedLenFeature(shape=4,dtype=tf.float32)
        })
    return out

if __name__ == '__main__':
    dataset = tf.data.TFRecordDataset(filenames = tf.io.matching_files('collected/*.tfrecord')).map(parse_example)
    plt.figure()
    es = []
    bis = []
    prop = []
    for i,example in enumerate(iter(dataset)):
        plt.plot(example['bis'])
        es.append(tf.math.reduce_sum(tf.abs(example['ref']-example['bis'])))
        bis.append(example['bis'])
        prop.append(example['prop'])
    plt.figure()
    idx = tf.argsort(es)
    for i in idx[:100]:
        plt.plot(bis[i])
    plt.figure()
    for i in idx[:100]:
        plt.plot(prop[i])
    plt.figure()
    plt.plot(bis[idx[0]])
    plt.plot(next(iter(dataset))['ref'])
    plt.figure()
    plt.plot(prop[idx[0]])
    plt.show()
