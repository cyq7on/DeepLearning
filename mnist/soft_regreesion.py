# code:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# 创建输入图片占位符，一张图片一个784的行向量表示，None表示可以输入任意张图片
x = tf.placeholder(tf.float32, [None, 784])
# 创建图片标签占位符
y_ = tf.placeholder(tf.float32, [None, 10])

# 创建权重矩阵变量
w = tf.Variable(tf.zeros([784, 10]))
# 创建偏移变量
b = tf.Variable(tf.zeros([10]))

# softmax回归 y=softmax(Wx + b)
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 随机梯度下降优化参数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建Session
session = tf.InteractiveSession()

# 初始化所有变量并运行
tf.global_variables_initializer().run()

# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
    # batch_xs, batch_ys对应着两个占位符x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行，并传入占位符的值
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax取出最大值的位置，比较预测值和标签值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 在Session中运行，计算准确率
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
