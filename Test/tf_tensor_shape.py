import tensorflow as tf
g1 = tf.random.Generator.from_seed(1)
a = g1.normal(shape=[16, 10, 29])

print(a[0, :, :])
print(a.shape)


b = tf.reshape(a, [-1, 29])
# print(b)
print(b.shape)

print(b[0, :])
print(b[1, :])
print(b[2, :])

c = tf.reshape(b, [16, 10, 29])
print(a[0, :, :])
print(c[0, :, :])


d = tf.reshape(a, [16, -1])
print(a[0, :, :])
print(d[0, :])