import tensorflow as tf 

FROZEN_DIR = './frozen_graph/'
FROZEN_FILE = 'graph.pb'

if __name__ == "__main__":
    # 直接加载计算图
    # parse the graph_def file

    with tf.gfile.GFile(FROZEN_DIR + FROZEN_FILE, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 

    # load the graph_def in the default graph

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "trained",   
            op_dict = None,   
            producer_op_list = None  
        )
        inputs = graph.get_tensor_by_name('trained/input:0')
        w = graph.get_tensor_by_name('trained/weights:0')
        b = graph.get_tensor_by_name('trained/bias:0') 

        infer_ret = tf.matmul(inputs, w) + b
        feed_dict = {
            inputs: [[84, 46]],
        }
    for op in graph.get_operations():  
        print(op.name,op.values()) 
    with tf.Session(graph=graph) as sess:
        print(sess.run(infer_ret, feed_dict))