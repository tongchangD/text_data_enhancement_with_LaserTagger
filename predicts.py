# Lint as: python3
"""Compute realized predictions for a dataset."""
# 参数 --input_file=./data/test.txt --input_format=wikisplit --output_file=./pred.tsv --label_map_file=./output/label_map.txt  --vocab_file=./data/model/RoBERTa-tiny-clue/vocab.txt --max_seq_length=40 --saved_model=./output/models/wikisplit_experiment_name/export/(ls "./output/models/wikisplit_experiment_name/export/" | grep -v "temp-" | sort -r | head -1)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import math, time
from termcolor import colored
import tensorflow as tf

import bert_example
import predict_utils
import tagging_converter
import utils

from curLine_file import curLine

FLAGS = flags.FLAGS

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
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')


def main_file(argv):
    print("argv",argv)
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
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
    print("FLAGS.vocab_file",FLAGS.vocab_file)
    print("FLAGS.max_seq_length",FLAGS.max_seq_length)
    print("FLAGS.do_lower_case",FLAGS.do_lower_case)
    print("converter",converter)
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, converter)

    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    sources_list = []
    # target_list = []
    with tf.io.gfile.GFile(FLAGS.input_file) as f:
        for line in f:
            sources = line.rstrip('\n')
            sources_list.append([sources])
            # target_list.append(target)
    number = len(sources_list)  # 总样本数
    predict_batch_size = min(64, number)
    batch_num = math.ceil(float(number) / predict_batch_size)

    start_time = time.time()
    num_predicted = 0
    with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
        writer.write(f'source\tprediction\ttarget\n')
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            print("sources_batch",sources_batch)
            prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
            print("prediction_batch",prediction_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
                #target = target_list[batch_id * predict_batch_size + id]
                writer.write(f'{"".join(sources)}\t\t{prediction}\n')

            if batch_id % 20 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    logging.info(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time / num_predicted} min.')
    print("done")

def main_sentence(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('label_map_file')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('saved_model')

    label_map = utils.read_label_map(FLAGS.label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        FLAGS.enable_swap_tag)
    print("FLAGS.vocab_file",FLAGS.vocab_file)
    print("FLAGS.max_seq_length",FLAGS.max_seq_length)
    print("FLAGS.do_lower_case",FLAGS.do_lower_case)
    print("converter",converter)
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, converter)

    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    # print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    # sources_list = []
    # target_list = []
    # with tf.io.gfile.GFile(FLAGS.input_file) as f:
    #     for line in f:
    #         sources = line.rstrip('\n')
    #         sources_list.append([sources])
    #         # target_list.append(target)
    while True:
        sentence=input(">> ")
        batch_num = 1
        start_time = time.time()
        num_predicted = 0
        for batch_id in range(batch_num):
            # sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            sources_batch=[sentence]
            prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
                # target = target_list[batch_id * predict_batch_size + id]
                print("原句sources: %s 拓展句predict: %s"%(sentence,prediction))
        # cost_time = (time.time() - start_time) / 60.0
        print("耗时",(time.time() - start_time) / 60.0,"s")
        # logging.info(
        #     f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time / num_predicted} min.')



if __name__ == '__main__':
    if input("files or sentence>> ") == "files":
        app.run(main_file)
    else:
        app.run(main_sentence)