import numpy as np
import tensorflow as tf
import datetime
import time

def _parse_dense_features(s, dshape, dtype=tf.float32, delimiter=','):
    record_defaults = [[0.0]] * dshape[1]
    value = tf.decode_csv(s, record_defaults=record_defaults, field_delim=delimiter)
    value = tf.stack(value, axis=1)
    value = tf.cast(value, dtype)
    return tf.reshape(value, dshape)

def _invert_permutation(input, row_count):
    '''wrapper for matrix'''
    rows = []
    for i in range(row_count):
        row = input[i,:]
        rows.append(tf.invert_permutation(row))
    return tf.cast(tf.stack(rows, axis=0), tf.float32)

def input_fn(name="input", tables="", num_epochs=None, num_workers=1, worker_id=0, capacity=0, batch_size=64):
    with tf.variable_scope(name_or_scope=name, reuse=False) as scope:
        with tf.device(device_name_or_function = ("/job:localhost/replica:0/task:%d"%worker_id) if worker_id != -1 else None):
            filename_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs)
            reader = tf.TextLineReader()
            keys, values = reader.read_up_to(filename_queue, batch_size)
            batch_keys, batch_values = tf.train.batch(
                [keys, values],
                batch_size=batch_size,
                capacity=10 * batch_size,
                enqueue_many=True,
                num_threads=1)
            record_defaults = [['']] * 4 + [[-1]] + [['']] * 9
            data = tf.decode_csv(batch_values, record_defaults=record_defaults, field_delim=';')

            pageid = data[4]
            ctr = data[7]
            cvr = data[8]
            price = data[9]
            isclick = data[10]
            pay = data[11]

            ctr = _parse_dense_features(ctr, (-1, 50))
            cvr = _parse_dense_features(cvr, (-1, 50))
            price = _parse_dense_features(price, (-1, 50))
            isclick = _parse_dense_features(isclick, (-1, 50))
            pay = _parse_dense_features(pay, (-1, 50))

            batch_data = {'keys': batch_keys,
                        'pageid': pageid,
                        'ctr': ctr,
                        'cvr': cvr,
                        'price': price,
                        'click': isclick,
                        'pay': pay}
    return batch_data