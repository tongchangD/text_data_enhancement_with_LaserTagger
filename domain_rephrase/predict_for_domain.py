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
import tensorflow as tf

block_list = os.path.realpath(__file__).split("/")
path = "/".join(block_list[:-2])
sys.path.append(path)

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

def remove_p(text_pre):
    text = text_pre[:9].replace('\n', '').replace('\r', '').replace('news', '').replace('food','').replace('GetPath', '') \
        .replace('GetYear','').replace('GetDate', '').replace('flight', '').replace('weather', '').replace('currency', '').replace('stock', '').replace('story', '') \
        .replace('drama', '').replace('jokes', '').replace('other','').replace('poetry', '').replace('GetLunar', '')+text_pre[9:]
    if len(text) == len(text_pre)  and len(text)>9:
        text = text.replace('stoytelling', '').replace('storytypes', '').replace(
            'SetPathPlace', '').replace('GetPoetryByTitle', '').replace('GetTitles', '') \
            .replace('GetSolarterm', '').replace('GetLastPhrases', '').replace('SelectPlaceIndex','').replace(
            'photo.tag', '').replace('photo.rac', '').replace('GetSuitAndAvoid', '') \
            .replace('GetOnePoetry', '').replace('SetPathTrans', '').replace('navigation', '').replace(
            'GetNextPhrases', '').replace('crosstalk', '')
    if len(text) == len(text_pre) and len(text) > 5:
        text = text.replace('currency,SetCurrentcy', '').replace('SetCurrentcy', '').replace('GetWeekDay', '').replace('GetAuthors', '') \
            .replace('trafficrestr', '').replace('GetTranslates', '').replace('GetAuthorNames','') \
            .replace('MatchTitle', '').replace('WeiboFirst', '').replace('music','').replace('times', '').replace('photo', '')
    return text

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
    predict_batch_size = 64
    batch_num = 0
    num_predicted = 0
    with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
        with open(FLAGS.input_file, "r") as f:
            sources_batch = []
            previous_line_list = []
            context_list = []
            line_number = 0
            start_time = time.time()
            while True:
                line_number +=1
                line = f.readline().rstrip('\n').strip("\"").strip(" ")
                if len(line) == 0:
                    break

                column_index = line.index(",")
                text = line[column_index+1:].strip("\"")  # context and query
                # for charChinese_id, char in enumerate(line[column_index+1:]):
                #     if (char>='a' and char<='z') or (char>='A' and char<='Z'):
                #         continue
                #     else:
                #         break
                source = remove_p(text)
                if source not in text: # TODO  ｉｇｎｏｒｅ的就给空字符串，这样输出也是空字符串
                    print(curLine(), "line_number=%d, ignore:%s" % (line_number, text), ",source:", len(source), source)
                    source = ""
                    # continue
                context_list.append(text[:text.index(source)])
                previous_line_list.append(line)
                sources_batch.append(source)
                if len(sources_batch) == predict_batch_size:
                    num_predicted, batch_num = predict_and_write(predictor, sources_batch,
                                     previous_line_list,context_list, writer, num_predicted, start_time, batch_num)
                    sources_batch = []
                    previous_line_list = []
                    context_list = []
                    # if num_predicted > 1000:
                    #     break
            if len(context_list)>0:
                num_predicted, batch_num = predict_and_write(predictor, sources_batch,
                                                             previous_line_list, context_list, writer, num_predicted,
                                                             start_time, batch_num)
    cost_time = (time.time() - start_time) / 60.0
    logging.info(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time / num_predicted/60} hours.')

if __name__ == '__main__':
    app.run(main)
