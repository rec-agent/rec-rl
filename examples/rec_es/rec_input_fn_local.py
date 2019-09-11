import numpy as np
import tensorflow as tf
import datetime
import time
import json

def _parse_dense_features(s, dshape, dtype=tf.float32, delimiter=','):
    record_defaults = [[0.0]] * dshape[1]
    #过decode_csv输出一个list of tensors，长度为50，每个item是一个float32类型的batch data
    value = tf.decode_csv(s, record_defaults=record_defaults, field_delim=delimiter)
    #stack将list里的tensor拼接成一个tensor
    value = tf.stack(value, axis=1)
    #进行数据类型转换
    value = tf.cast(value, dtype)
    return tf.reshape(value, dshape)

def _parse_dense_int_features(s, dshape, dtype=tf.int64, delimiter=','):
    record_defaults = [[0.0]] * dshape[1]
    #过decode_csv输出一个list of tensors，长度为50，每个item是一个float32类型的batch data
    value = tf.decode_csv(s, record_defaults=record_defaults, field_delim=delimiter)
    #stack将list里的tensor拼接成一个tensor
    value = tf.stack(value, axis=1)
    #进行数据类型转换
    value = tf.cast(value, dtype)
    return tf.reshape(value, dshape)

def _invert_permutation(input, row_count):
    '''wrapper for matrix'''
    rows = []
    for i in range(row_count):
        row = input[i,:]
        rows.append(tf.invert_permutation(row))
    return tf.cast(tf.stack(rows, axis=0), tf.float32)

def input_fn(spu2item_dict=None, name="input", tables="", num_epochs=None, num_workers=1, worker_id=0, capacity=0, batch_size=64):
    print('total num of spu ids:',len(spu2item_dict))
    with tf.variable_scope(name_or_scope=name, reuse=False) as scope:
        with tf.device(device_name_or_function = ("/job:localhost/replica:0/task:%d"%worker_id) if worker_id != -1 else None):
            filename_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs,shuffle=False)
            reader = tf.TextLineReader()
            keys, values = reader.read_up_to(filename_queue, batch_size)
            batch_keys, batch_values = tf.train.batch(
                [keys, values],
                batch_size=batch_size,
                capacity=10 * batch_size,
                enqueue_many=True,
                num_threads=1)
            # print(batch_keys)
            # print(batch_values)

            record_defaults = [[0.0]] * 5
            #将一行或多行数据读入，按‘;’进行分割，然后赋值为一个list of tensors，长度符合数据格式的列
            #record_defaults设定了每一列tensor的type，此处为前四个tensor为string，第五个为int32，六到十四继续是string
            data = tf.decode_csv(batch_values, record_defaults=record_defaults, field_delim=',')
            # print(data)

            #按列赋值给page_id,ctr,cvr等等
            user_id = tf.cast(data[0],tf.int64)
            page_num = tf.cast(data[1],tf.int64)
            item_id = tf.cast(data[2],tf.int64)
            spu = tf.cast(data[3],tf.int64)
            seller_id = tf.cast(data[4],tf.int64)
            
            with tf.Session() as sess:
                s = sess.run(spu)
            spu_info = spu2item_dict[str(s)]
            # key,value=list(zip(*spu2item_dict.items()))
#             # with tf.Session() as sess:
#             #     spu = sess.run(spu)
#             # spu = [i for i, s in enumerate(key) if s in spu]
#             # iteminfo = tf.gather(value,spu)

            batch_data = {'keys': batch_keys,
                        'item_id':item_id,
                        'page_num':page_num,
                        'user_id':user_id,
                        'spu':spu,
                        'seller_id':seller_id
                        # 'spu_info':spu_info
                        }

    return batch_data

# def input_fn(spu_dict=None, name="input", tables="", num_epochs=None, num_workers=1, worker_id=0, capacity=0, batch_size=64):
#     # print(len(spu_dict))
#     with tf.variable_scope(name_or_scope=name, reuse=False) as scope:
#         with tf.device(device_name_or_function = ("/job:localhost/replica:0/task:%d"%worker_id) if worker_id != -1 else None):
#             filename_queue = tf.train.string_input_producer(tables, num_epochs=num_epochs,shuffle=False)
#             reader = tf.TextLineReader()
#             keys, values = reader.read_up_to(filename_queue, batch_size)
#             batch_keys, batch_values = tf.train.batch(
#                 [keys, values],
#                 batch_size=batch_size,
#                 capacity=10 * batch_size,
#                 enqueue_many=True,
#                 num_threads=1)
#             # print(batch_keys)
#             # print(batch_values)

#             # record_defaults = [['']] * 4 + [[-1]] + [['']] * 9
#             record_defaults = [['']] * 2 + [[-1]] + [['']] * 32
#             #将一行或多行数据读入，按‘;’进行分割，然后赋值为一个list of tensors，长度符合数据格式的列
#             #record_defaults设定了每一列tensor的type，此处为前四个tensor为string，第五个为int32，六到十四继续是string
#             data = tf.decode_csv(batch_values, record_defaults=record_defaults, field_delim=';')
#             # print(data)

#             #按列赋值给page_id,ctr,cvr等等
#             # pageid = data[4] #type: int32
#             # ctr = data[7] #type: string
#             # cvr = data[8] #type: string
#             # price = data[9] #type: string
#             # isclick = data[10] #type: string
#             # pay = data[11] #type: string
#             # c = np.random.random([10,1]) 
#             # pageid = tf.nn.embedding_lookup(c, data[2])  

#             pageid = data[2] #type: int32
#             spu=data[31]
#             itemid = data[3]
#             ctr = data[8] #type: string
#             cvr = data[9] #type: string
#             price = data[10] #type: string
#             isclick = data[33] #type: string
#             pay = data[34] #type: string
            
#             #ctr,cvr等实际上是string，里面每行包含了50个item，需要通过_parse_dense_features转成float形式
#             itemid = _parse_dense_int_features(itemid,(-1,50))
#             spu = _parse_dense_int_features(spu,(-1,50))
#             # for i in itemid:

#             ctr = _parse_dense_features(ctr, (-1, 50)) #batch_size x 50
#             cvr = _parse_dense_features(cvr, (-1, 50))
#             price = _parse_dense_features(price, (-1, 50))
#             isclick = _parse_dense_features(isclick, (-1, 50))

#             # isclick = tf.cast(tf.nn.embedding_lookup(c, _parse_dense_int_features(isclick, (-1, 50))),tf.float32)
#             # isclick = tf.reshape(isclick,(-1,50))

#             pay = _parse_dense_features(pay, (-1, 50))
#             # print(ctr)

#             # key,value=list(zip(*spu_dict.items()))
#             # with tf.Session() as sess:
#             #     spu = sess.run(spu)
#             # spu = [i for i, s in enumerate(key) if s in spu]
#             # iteminfo = tf.gather(value,spu)



#             batch_data = {'keys': batch_keys,
#                         'itemid':itemid,
#                         # 'iteminfo':iteminfo,
#                         'pageid': pageid,
#                         'ctr': ctr,
#                         'cvr': cvr,
#                         'price': price,
#                         'click': isclick,
#                         'pay': pay}

#     return batch_data


if __name__ == '__main__':
    print('loading spu2item dict...')
    f=open("./dataset/spu2item.json","r")
    for line in f:
        spu2item_dict=json.loads(line)
    f.close()
    # print(spu_dict)


    # tables = ["padded_pv_1k.data"]
    tables = ["./dataset/train_pv.data"]
    batch_data = input_fn(
                spu2item_dict=spu2item_dict,
                name='table_env',
                tables=tables,
                #num_epochs：一个整数（可选）。
                #如果指定，string_input_producer在产生OutOfRange错误之前从string_tensor中产生num_epochs次字符串。
                #如果未指定，则可以无限次循环遍历字符串。
                # num_epochs=None,
                num_epochs=1,
                num_workers=1,
                worker_id=0,
                capacity=10000,
                batch_size=1       
                
            )
    

   

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord=tf.train.Coordinator()
        #train.start_queue_runners()函数才会启动填充队列的线程，系统不再“停滞”，此后计算单元就可以拿到数据并进行计算
        thread=tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():
                sess.run(batch_data)
                # print(batch_data['spu_info'].eval())
                print(batch_data['item_id'].shape)
        #当文件队列读到末尾的时候，抛出异常
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()#将读取文件的线程关闭
        coord.join(thread)#将读取文件的线程加入到主线程中（虽然说已经关闭过）