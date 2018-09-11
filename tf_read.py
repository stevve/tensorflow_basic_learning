import os
import tensorflow as tf


#批处理只是每次取多少个数，跟队列，和数据数量没有影响
#tesorflow csv读入函数
def csvRead(filelist):
    #构造队列
    file_queue = tf.train.string_input_producer(filelist)

    # 构造csv阅读器，读取一行数据
    reader = tf.TextLineReader()
    # 对每行内容解码，record_default :指定每一个样本的每一列的类型，指定默认值为[['None'],[4.0]]
    records = [['None'], ['None']]
    key, value = reader.read(file_queue)

    #批处理
    example, label = tf.decode_csv(value, record_defaults=records)

    example_batch,label_batch=tf.train.batch([example,label],batch_size=9,num_threads=1,capacity=20)

    return example_batch,label_batch
#tensorflow 图片读入函数
def picRead(filelist):
    """

    :param filelist:
    :return: 每张图片张量
    """
    file_queue=tf.train.string_input_producer(filelist)

    #2.构造读取器内容
    reader=tf.WholeFileReader()
    key,value=reader.read(file_queue)
    #对读取的内容解码
    image=tf.image.decode_jpeg(value)

    #5.统一图片大小
    image_resize=tf.image.resize_images(image,[200,200])
    image_resize.set_shape([200,200,3])

    #批处理图片
    image_batch=tf.train.batch([image_resize],batch_size=20,num_threads=1,capacity=20)

    return image_batch
#定义cifar 数据的命令行参数
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cifar_dir','')


class CifarRead():
    """
    完成二进制文件读取，写进tfrecords,读取tfrecords

    """
    def __init__(self,filelist):
        self.filelist=filelist

    def read_and_decode(self):
       #构造文件队列
        file_queue=tf.train.string_input_producer(self.filelist)
    #2.构造二进制文件读取器，每个样本的字节数
        reader=tf.FixedLengthRecordReader(record_bytes=3073)
        key,value=reader.read(file_queue)
        #3二进制解码
        label_iamge=tf.decode_raw(value,tf.uint8)
        #分割出图片和标签
        label=tf.cast(tf.slice(label_iamge,[0],[3073]),tf.int32)
        image=tf.slice(label_iamge,[3073],[3073])

        image_reshape=tf.reshape(image,[32,32,3])

        #批处理数据
        image_batch,label_batch=tf.train.batch([image_reshape,label],batch_size=10,capacity=10)
        return image_batch,label_batch
if __name__ == '__main__':
    file_name = os.listdir("./images/")
    # filelist = [os.path.join("./csv/", file) for file in file_name]
    filelist = [os.path.join("./images/", file) for file in file_name]
    print(filelist)

    # example_batch, label_batch=csvRead(filelist)
    image_resize=picRead(filelist)

    with tf.Session() as sess:
        #定义一个线程协调器
        coord=tf.train.Coordinator()
        # 开启读取文件的线程
        thread=tf.train.start_queue_runners(sess,coord=coord)

        # print(sess.run([example_batch,label_batch]))
        print(sess.run([image_resize]))
        #回收子线程
        coord.request_stop()
        coord.join(thread)

