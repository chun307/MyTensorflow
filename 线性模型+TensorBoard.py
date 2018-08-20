import tensorflow as tf

SUMMARY_DIR = "./log"

# 创建变量 W 和 b 节点，并设置初始值
with tf.variable_scope('W'):
    W = tf.Variable([0], dtype=tf.float32, name="W")
    # 这样在TB种就能看到W的训练曲线了by chun
    tf.summary.histogram("W", W)

with tf.variable_scope('b'):
    b = tf.Variable([0], dtype=tf.float32, name="b")
    # 这样在TB种就能看到b的训练曲线了by chun
    tf.summary.histogram("b", b)

# 创建 x 节点，用来输入实验中的输入数据
with tf.name_scope('input_x'):
    x = tf.placeholder(tf.float32)
# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
with tf.name_scope('input_y'):
    y = tf.placeholder(tf.float32)
# 创建线性模型
with tf.name_scope('output'):
    linear_model = W * x + b

# 创建损失模型
with tf.name_scope('loss'):
    loss = tf.reduce_sum(tf.square(linear_model - y))
    # 必须至少创建一个summary，下面第54行才不会出错by chun
    tf.summary.scalar("loss", loss)

# 创建一个梯度下降优化器，学习率为0.001
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.001)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)

# 用两个数组保存训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

# 创建 Session 用来计算模型
sess = tf.Session()
with tf.name_scope('init'):
    init = tf.global_variables_initializer()
sess.run(init)

# 创建一个日志文件merged，来存储所有的summary
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

# 训练10000次
for i in range(10000):
    summary, _ = sess.run([merged, train], {x: x_train, y: y_train})
    summary_writer.add_summary(summary, i)
    if i % 100 == 0:
        print('After %d W: %s b: %s loss: %s' % (i, sess.run(W), sess.run(b), sess.run(loss, {x: x_train, y: y_train})))

summary_writer.close()
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: x_train, y: y_train})))
