# coding=utf-8
# 为ｄｏｍａｉｎ识别泛化语料

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import os, sys, time
from termcolor import colored
import math
import tensorflow as tf

block_list = os.path.realpath(__file__).split("/")
path = "/".join(block_list[:-2])
sys.path.append(path)
from compute_lcs import _compute_lcs
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

def predict_and_write(predictor, sources_batch, previous_line_list,context_list, writer, num_predicted, start_time, batch_num):
    prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
    assert len(prediction_batch) == len(sources_batch)
    for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
        output = ""
        if len(prediction) > 1 and prediction != sources:  #  TODO  ignore keep totelly and short prediction
            output= "%s%s" % (context_list[id], prediction)  # 需要和ｃｏｎｔｅｘｔ拼接么
            # print(curLine(), "prediction:", prediction, "sources:", sources, ",output:", output, prediction != sources)
        writer.write("%s\t%s\n" % (previous_line_list[id], output))
    batch_num = batch_num + 1
    num_predicted += len(prediction_batch)
    if batch_num % 200 == 0:
        cost_time = (time.time() - start_time) / 3600.0
        print("%s batch_id=%d, predict %d examples, cost %.3fh." %
              (curLine(), batch_num, num_predicted, cost_time))
    return num_predicted, batch_num


def main(argv):
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
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, converter)
    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    sourcesA_list = []
    sourcesB_list = []
    target_list = []
    with tf.io.gfile.GFile(FLAGS.input_file) as f:
        for line in f:
            sourceA, sourceB, label = line.rstrip('\n').split('\t')
            sourcesA_list.append([sourceA.strip(".")])
            sourcesB_list.append([sourceB.strip(".")])
            target_list.append(label)



    number = len(sourcesA_list)  # 总样本数
    predict_batch_size = min(32, number)
    batch_num = math.ceil(float(number) / predict_batch_size)

    start_time = time.time()
    num_predicted = 0
    prediction_list = []
    with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
        for batch_id in range(batch_num):
            sources_batch = sourcesA_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            batch_b = sourcesB_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            location_batch = []
            sources_batch.extend(batch_b)
            for source in sources_batch:
                location = list()
                for char in source[0]:
                    if (char>='0' and char<='9') or char in '.- ' or (char>='a' and char<='z') or (char>='A' and char<='Z'):
                        location.append("1") # TODO TODO
                    else:
                        location.append("0")
                location_batch.append("".join(location))
            prediction_batch = predictor.predict_batch(sources_batch=sources_batch, location_batch=location_batch)
            current_batch_size = int(len(sources_batch)/2)
            assert len(prediction_batch) == current_batch_size*2

            for id in range(0, current_batch_size):
                target = target_list[num_predicted+id]
                prediction_A = prediction_batch[id]
                prediction_B = prediction_batch[current_batch_size+id]
                sourceA = "".join(sources_batch[id])
                sourceB = "".join(sources_batch[current_batch_size + id])
                if prediction_A == prediction_B: # 其中一个换为source
                    lcsA = len(_compute_lcs(sourceA, prediction_A))
                    if lcsA < 8: # A的变化大
                        prediction_B = sourceB
                    else:
                        lcsB = len(_compute_lcs(sourceB, prediction_B))
                        if lcsA <= lcsB:  # A的变化大
                            prediction_B = sourceB
                        else:
                            prediction_A = sourceA
                            print(curLine(), batch_id, prediction_A, prediction_B, "target:", target, "current_batch_size=",
                                current_batch_size, "lcsA=%d,lcsB=%d" % (lcsA, lcsB))
                writer.write(f'{prediction_A}\t{prediction_B}\t{target}\n')

                prediction_list.append("%s\t%s\n"% (sourceA, prediction_A))
                # print(curLine(), id,"sourceA:", sourceA, "sourceB:",sourceB, "target:", target)
                prediction_list.append("%s\t%s\n" % (sourceB, prediction_B))
            num_predicted += current_batch_size
            if batch_id % 20 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print(curLine(), id, prediction_A, prediction_B, "target:", target, "current_batch_size=", current_batch_size)
                print(curLine(), id,"sourceA:", sourceA, "sourceB:",sourceB, "target:", target)
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    with open("prediction.txt", "w") as prediction_file:
        prediction_file.writelines(prediction_list)
        print(curLine(), "save to prediction_qa.txt.")
    cost_time = (time.time() - start_time) / 60.0
    print(curLine(), id, prediction_A, prediction_B, "target:", target, "current_batch_size=", current_batch_size)
    print(curLine(), id, "sourceA:", sourceA, "sourceB:", sourceB, "target:", target)
    logging.info(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time / num_predicted*60000}ms.')

if __name__ == '__main__':
    app.run(main)
