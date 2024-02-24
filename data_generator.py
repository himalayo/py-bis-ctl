import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from keras.models import Model, load_model
import patient
import controllers
import json
import gc
import time

rfs = tf.concat([[0.5]*200,[0.3]*200,[0.5]*200,[0.65]*200,[0.5]*200,[0.4]*200],0)

@tf.function
def generate_data(mdl,z):
    k = controllers.swarm_PID(mdl,0.5,z)
    return controllers.run_pid(k[0],k[1],k[2], rfs,mdl,z)

@tf.function
def collect(mdl,zs,p_iters=5):
    return tf.map_fn(tf.function(lambda z: generate_data(mdl,z)),zs,parallel_iterations=p_iters)

def run():
    mdl = load_model('./weights')
    zs = tf.random.normal((50,1,4))
    data = collect(mdl,zs,p_iters=10) 
    with tf.io.TFRecordWriter(f"./collected/{tf.random.uniform((),maxval=int(1e8),dtype=tf.int32)}.tfrecord") as writer:
        for i in range(zs.shape[0]):
            ex = tf.train.Example(features=tf.train.Features(feature={
                'ref': tf.train.Feature(float_list=tf.train.FloatList(value=rfs.numpy())),
                'bis': tf.train.Feature(float_list=tf.train.FloatList(value=data[i,0].numpy())),
                'remi': tf.train.Feature(float_list=tf.train.FloatList(value=data[i,1].numpy())),
                'prop': tf.train.Feature(float_list=tf.train.FloatList(value=data[i,2].numpy())),
                'z': tf.train.Feature(float_list=tf.train.FloatList(value=zs[i,-1].numpy()))
                })).SerializeToString()
            writer.write(ex)
