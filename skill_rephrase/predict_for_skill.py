# coding=utf-8
# 为任务型语料做泛化　意图和槽位识别

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import math, json
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
  num_predicted = 0

  sources_list = []
  location_list = []
  corpus_id_list = []
  entity_list = []
  domainname_list = []
  intentname_list = []
  context_list = []
  template_id_list = []
  with open(FLAGS.input_file, "r") as f:
      corpus_json_list = json.load(f)
      # corpus_json_list = corpus_json_list[:100]
      for corpus_json in corpus_json_list:
          sources_list.append([corpus_json["oriText"]])
          location_list.append(corpus_json["location"])
          corpus_id_list.append(corpus_json["corpus_id"])
          entity_list.append(corpus_json["entity"])
          domainname_list.append(corpus_json["domainname"])
          intentname_list.append(corpus_json["intentname"])
          context_list.append(corpus_json["context"])
          template_id_list.append(corpus_json["template_id"])
  number = len(sources_list)  # 总样本数
  predict_batch_size = min(64, number)
  batch_num = math.ceil(float(number) / predict_batch_size)
  start_time = time.time()
  index = 0
  for batch_id in range(batch_num):
      sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
      location_batch = location_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
      prediction_batch = predictor.predict_batch(sources_batch=sources_batch, location_batch=location_batch)
      assert len(prediction_batch) == len(sources_batch)
      num_predicted += len(prediction_batch)
      for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
          index = batch_id * predict_batch_size + id
          output_json = {"corpus_id": corpus_id_list[index], "oriText": prediction, "sources": sources[0],
                         "entity": entity_list[index], "location": location_list[index],
                         "domainname": domainname_list[index], "intentname": intentname_list[index],
                         "context": context_list[index], "template_id": template_id_list[index]}
          corpus_json_list[index]= output_json
      if batch_id % 20 == 0:
          cost_time = (time.time() - start_time) / 60.0
          print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
  assert len(corpus_json_list) == index + 1
  with open(FLAGS.output_file, 'w', encoding='utf-8') as writer:
    json.dump(corpus_json_list, writer, ensure_ascii=False, indent=4)
  cost_time = (time.time() - start_time) / 60.0
  logging.info(f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time/num_predicted} min.')


if __name__ == '__main__':
  app.run(main)
