#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NameDataSet = 'name.csv'

train_x = []
train_y = []

with open(NameDataSet, 'r', encoding='UTF-8') as f:
    first_line = True
    for line in f:
        if first_line is True:
            first_line = False
            continue
        sample = line.strip().split(',')    # strip:去掉首尾空格; split(','):从','处分开字符串
        if len(sample) == 2:                # sample = ['李健虎', '男']
            train_x.append(sample[0])
            if sample[1] == '男':
                train_y.append([0, 1])  # 男
            else:
                train_y.append([1, 0])  # 女

# max_name_length = max([len(name) for name in train_x])
# print("最长名字的字符数: %s" % max_name_length)
max_name_length = 8

# 词汇表（参看聊天机器人练习）
counter = 0
vocabulary = {}
for name in train_x:
    counter += 1
    tokens = [word for word in name]
    for word in tokens:
        if word in vocabulary:
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

# print(vocabulary) # {'熊': 1946, '猫': 5, '哥': 20, '周': 2169, '笑': 545,...}
vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)  # 将名字中出现的汉字进行按使用频率排序 共有6018个汉字，第一个为空格
# print(vocabulary_list)  # [' ', '子', '雨', '文', '涵', '宇',
'''
sorted用法：按reverse=True降序排列
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=lambda s: s[2])            # 按年龄排序
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
'''
# print(len(vocabulary_list))

# 字符串转为向量形式
vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
# print(vocab)  # {' ': 0, '子': 1, '雨': 2, '文': 3, '涵': 4,...}
'''
enumerate用法：遍历的数据，同时列出数据和数据下标
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 小标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
'''
train_x_vec = []
for name in train_x:
    name_vec = []
    for word in name:
        name_vec.append(vocab.get(word))
    while len(name_vec) < max_name_length:
        name_vec.append(0)
    train_x_vec.append(name_vec)
# print(train_x_vec)  # 统计name.csv中的每个姓名汉字出现的频率[628, 1763, 793, 0, 0, ], [88, 24, 36, 0, 0,],...[名字第一个字出现次数，第二个出现次数，第三个，四，五...]
input_size = max_name_length
num_classes = 2

batch_size = 64
num_batch = len(train_x_vec) // batch_size   # //是取整数的运算

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

dropout_keep_prob = tf.placeholder(tf.float32)


def neural_network(vocabulary_size, embedding_size=128, num_filters=128):
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    # convolution + maxpool layer
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    # h_pool = tf.concat(3, pooled_outputs)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes])
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)

    return output


# 训练
def train_neural_network():
    output = neural_network(len(vocabulary_list))

    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(201):
            for i in range(num_batch):
                batch_x = train_x_vec[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
                print("%s %s %s" % (e, i, loss_))
            if e % 50 == 0:
                saver.save(sess, "./name2sex.model", global_step=e)


# train_neural_network()


# 使用训练的模型
def detect_sex(name_list):
    x = []
    for name in name_list:
        NameVec = []
        for word in name:
            NameVec.append(vocab.get(word))
        while len(NameVec) < max_name_length:
            NameVec.append(0)
        x.append(NameVec)

    output = neural_network(len(vocabulary_list))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt is not None:
            # print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        predictions = tf.argmax(output, 1)
        res = sess.run(predictions, {X: x, dropout_keep_prob: 1.0})

        i = 0
        for name in name_list:
            print(name, '女' if res[i] == 0 else '男')
            i += 1

    writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    writer.close()


if __name__ == '__main__':
    InputName = input("请输入姓名：")
    detect_sex([InputName])
    input("\n输入回车结束")

    # while 1:
    # print("你输入的内容是: ", InputName)
    # detect_sex(["白富美", "高帅富", "王婷婷", "田野", "白百合", "李健"])
