# -*- coding: utf-8 -*-
#调用服务
import time, math
import json
import requests
import tensorflow as tf
from curLine_file import curLine

user_name="wzk"
test_num_max=300000
language='Chinese'
# 此URL中IP地址为参赛者的服务器地址，应为可达的公网IP地址，端口默认21628
url = "http://0.0.0.0:6000/rephrase"

# 调用ＡＰＩ接口
def http_post(sources_batch):
    parameter = {'text_list': sources_batch}
    headers = {'Content-type': 'application/json'}
    status = -1
    output = None
    try:
        r = requests.post(url, data=json.dumps(
            parameter), headers=headers, timeout=10.5)
        if r.status_code == 200:
            result = r.json()
            # print(curLine(),result)
            status = result['status']
            version = result['version']
            if status == 0:
                data = result["data"]
                output = data['output']
            else:
                print(curLine(), "version:%s, status=%d, message:%s" % (version, status, result['message']))
        else:
            print("%sraise wrong,status_code: " % (curLine()), r.status_code)
    except Exception as e:
        print(curLine(), Exception, ' : ', e)
        input(curLine())
    return status, output

def test():
    """
    此函数为测试函数，将sh运行在服务器端后，用该程序在另一网络测试
    This function is a test function.
    Run this function for test in a network while ServerDemo.py is running on a server in a different network
    """
    sources_list = []
    target_list = []
    output_file = "/home/cloudminds/Mywork/corpus/rephrase_corpus/pred.tsv"
    input_file= "/home/cloudminds/Mywork/corpus/rephrase_corpus/test.txt"
    with tf.io.gfile.GFile(input_file) as f:
      for line in f:
          sources, target, lcs_rate = line.rstrip('\n').split('\t')
          sources_list.append(sources) # [sources])
          target_list.append(target)
    number = len(target_list)  # 总样本数
    predict_batch_size = min(64, number) # TODO
    batch_num = math.ceil(float(number)/predict_batch_size)
    num_predicted = 0
    with open(output_file, 'w') as writer:
        writer.write(f'source\tprediction\ttarget\n')
        start_time = time.time()
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1)*predict_batch_size]
            # prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
            status, prediction_batch = http_post(sources_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id,[prediction,sources] in enumerate(zip(prediction_batch, sources_batch)):
                target = target_list[batch_id * predict_batch_size + id]
                writer.write(f'{"".join(sources)}\t{prediction}\t{target}\n')
            if batch_id % 20 == 0:
                cost_time = (time.time()-start_time)/60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id+1, batch_num, num_predicted,number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    print(curLine(), "%d predictions saved to %s, cost %f min, ave %f min."
          % (num_predicted, output_file, cost_time, cost_time / num_predicted))


if __name__ == '__main__':
    rougeL_ave=test()
