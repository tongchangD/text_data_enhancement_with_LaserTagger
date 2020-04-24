# 文本复述服务 基于tensorflow框架
import os, sys
from absl import flags
from absl import logging
import numpy as np
import json
import logging
# http 接口
from flask import Flask, jsonify, request
app = Flask(__name__)

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

while logger.hasHandlers():
    for i in logger.handlers:
        logger.removeHandler(i)

user_name = ""  # wzk/
version="1.0.0.0"


block_list = os.path.realpath(__file__).split("/")
path = "/".join(block_list[:-2])
sys.path.append(path)


import bert_example
import predict_utils
import tagging_converter
import utils
import tensorflow as tf
# FLAGS = flags.FLAGS
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the TSV file where the predictions are written to.')
flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 40, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')

flags.DEFINE_string('host', None, 'host address.')
flags.DEFINE_integer('port', None, 'port.')


class RequestHandler():
    def __init__(self):
        # if len(argv) > 1:
        #     raise app.UsageError('Too many command-line arguments.')
        flags.mark_flag_as_required('input_file')
        flags.mark_flag_as_required('input_format')
        flags.mark_flag_as_required('output_file')
        flags.mark_flag_as_required('label_map_file')
        flags.mark_flag_as_required('vocab_file')
        flags.mark_flag_as_required('saved_model')

        label_map = utils.read_label_map(FLAGS.label_map_file)
        converter = tagging_converter.TaggingConverter(
            tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
            FLAGS.enable_swap_tag)
        builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                                  FLAGS.max_seq_length,
                                                  FLAGS.do_lower_case, converter)
        self.predictor = predict_utils.LaserTaggerPredictor(
            tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
            label_map)

    def infer(self, sources_batch):
        prediction_batch = self.predictor.predict_batch(sources_batch=sources_batch)
        assert len(prediction_batch) == len(sources_batch)

        return prediction_batch

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# http POST 接口
@app.route('/rephrase', methods=['POST'])  # 推理
def returnPost():
    data_json = {'version': version}
    try:
        question_raw_batch = request.json['text_list']
        # question_raw = question_raw_list[0]
    except TypeError:
        data_json['status'] = -1
        data_json['message'] = "Fail:input information type error"
        data_json['data'] = []
        return json.dumps(data_json, cls=MyEncoder, ensure_ascii=False)
    question_batch = []
    for question_raw in question_raw_batch:
        question_batch.append([question_raw.strip()])
    # logger.info('question: %s' % (question_raw_batch))
    # if len(question_raw) > FLAGS.max_seq_length*3:
    #     data_json['status'] = -1
    #     data_json['message'] = "Fail:the length of the input question must not exceed %d"% FLAGS.max_seq_length*3
    #     data_json['data'] = []
    #     print(curLine(), "status=%d, message:%s" % (data_json['status'],data_json['message']))
    #     return json.dumps(data_json, cls=MyEncoder, ensure_ascii=False)
    # if not question_raw or len(question_raw) == 0:
    #     data_json['status'] = -1
    #     data_json['message'] = "Fail:input information illegal"
    #     data_json['data'] = []
    #     print(curLine(), "status=%d, message:%s" % (data_json['status'],data_json['message']))
    #     return json.dumps(data_json, cls=MyEncoder, ensure_ascii=False)
    decoded_output = rHandler.infer(question_batch)
    if len(decoded_output) == 0:
        data_json['status'] = -1
        data_json['message'] = "Fail: fail to get the summary of the article."
        data_json['data'] = []
        return json.dumps(data_json, cls=MyEncoder, ensure_ascii=False)
    data_json['status'] = 0
    data_json['message'] = "Success"
    data_json['data'] = {}
    data_json['data']['output'] = decoded_output
    # endtime = datetime.datetime.now()
    # print('time:', (endtime - starttime).microseconds)
    return json.dumps(data_json, cls=MyEncoder, ensure_ascii=False)

rHandler = RequestHandler()

if __name__ == '__main__':


    app.run(host=FLAGS.host, port=FLAGS.port)
