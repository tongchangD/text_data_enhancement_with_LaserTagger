# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Optimizes the vocabulary of phrases that can be added by LaserTagger.

The goal is to find a fixed-size set of phrases that cover as many training
examples as possible. Based on the phrases, saves a file containing all possible
tags to be predicted and another file reporting the percentage of covered
training examples with different vocabulary sizes.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from typing import Sequence, Text

from absl import app
from absl import flags
from absl import logging

import utils

import numpy as np
import scipy.sparse
import tensorflow as tf
from compute_lcs import _compute_lcs
from curLine_file import curLine
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing source-target pairs from which the '
    'vocabulary is optimized (see `input_format` flag and utils.py for '
    'documentation).')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the `input_file`. See utils.py for '
    'documentation on the different formats.')
flags.DEFINE_integer(
    'max_input_examples', 50000,
    'At most this many examples from the `input_file` are used for optimizing '
    'the vocabulary.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the resulting file with all possible tags. Coverage numbers will '
    'be written to a separate file which has the same path but ".log" appended '
    'to it.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_integer('vocabulary_size', 500,
                     'Number of phrases to include in the vocabulary.')
flags.DEFINE_integer(
    'num_extra_statistics', 100,
    'Number of extra phrases that are not included in the vocabulary but for '
    'which we compute the coverage numbers. These numbers help determining '
    'whether the vocabulary size should have been larger.')





def _get_added_phrases(source: Text, target: Text) -> Sequence[Text]:
  """Computes the phrases that need to be added to the source to get the target.

  This is done by aligning each token in the LCS to the first match in the
  target and checking which phrases in the target remain unaligned.

  TODO(b/142853960): The LCS tokens should ideally be aligned to consecutive(连续不断的)
  target tokens whenever possible, instead of aligning them always to the first
  match. This should result in a more meaningful phrase vocabulary with a higher
  coverage.

  Note that the algorithm is case-insensitive and the resulting phrases are
  always lowercase.

  Args:
    source: Source text.
    target: Target text.

  Returns:
    List of added phrases.
  """
  # 英文是分成ｗｏｒｄ sep=' '，中文是分成字 sep=''
  sep = ''
  source_tokens = utils.get_token_list(source.lower()) # list(source.lower()) #
  target_tokens = utils.get_token_list(target.lower()) # list(target.lower()) #

  kept_tokens = _compute_lcs(source_tokens, target_tokens)
  added_phrases = []
  # Index of the `kept_tokens` element that we are currently looking for.
  kept_idx = 0
  phrase = []
  for token in target_tokens:
    if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
      kept_idx += 1
      if phrase:
        added_phrases.append(sep.join(phrase))
        phrase = []
    else:
      phrase.append(token)
  if phrase:
    added_phrases.append(sep.join(phrase))
  return added_phrases


def _added_token_counts(data_iterator, try_swapping, max_input_examples=10000):
  """Computes how many times different phrases have to be added.

  Args:
    data_iterator: Iterator to yield source lists and targets. See function
      yield_sources_and_targets in utils.py for the available iterators. The
      strings in the source list will be concatenated, possibly after swapping
      their order if swapping is enabled.
    try_swapping: Whether to try if swapping sources results in less added text.
    max_input_examples: Maximum number of examples to be read from the iterator.

  Returns:
    Tuple (collections.Counter for phrases, added phrases for each example).
  """
  phrase_counter = collections.Counter()
  num_examples = 0
  all_added_phrases = []
  max_seq_length = 0
  for sources, target in data_iterator:
    # sources 可能是多句话，后面用空格拼接起来
    if num_examples >= max_input_examples:
      break
    source_merge = ' '.join(sources)
    if len(source_merge) > max_seq_length:
        print(curLine(), "max_seq_length=%d, len(source_merge)=%d,source_merge:%s" %
              (max_seq_length, len(source_merge), source_merge))
        max_seq_length = len(source_merge)
    logging.log_every_n(logging.INFO, f'{num_examples} examples processed.', 10000)
    added_phrases = _get_added_phrases(source_merge, target)
    if try_swapping and len(sources) == 2:
      added_phrases_swap = _get_added_phrases(' '.join(sources[::-1]), target)
      # If we can align more and have to add less after swapping, we assume that
      # the sources would be swapped during conversion.
      if len(''.join(added_phrases_swap)) < len(''.join(added_phrases)):
        added_phrases = added_phrases_swap
    for phrase in added_phrases:
      phrase_counter[phrase] += 1
    all_added_phrases.append(added_phrases)
    num_examples += 1
  logging.info(f'{num_examples} examples processed.\n')
  return phrase_counter, all_added_phrases, max_seq_length


def _construct_added_phrases_matrix(all_added_phrases, phrase_counter):
  """Constructs a sparse phrase occurrence matrix.

  Examples are on rows and phrases on columns.

  Args:
    all_added_phrases: List of lists of added phrases (one list per example).
    phrase_counter: Frequence of each unique added phrase.

  Returns:
    Sparse boolean matrix whose element (i, j) indicates whether example i
    contains the added phrase j. Columns start from the most frequent phrase.
  """
  phrase_2_idx = {
      tup[0]: i for i, tup in enumerate(phrase_counter.most_common())
  }
  matrix = scipy.sparse.dok_matrix((len(all_added_phrases), len(phrase_2_idx)),
                                   dtype=np.bool)
  for i, added_phrases in enumerate(all_added_phrases):
    for phrase in added_phrases:
      phrase_idx = phrase_2_idx[phrase]
      matrix[i, phrase_idx] = True
  # Convert to CSC format to support more efficient column slicing.
  return matrix.tocsc()


def _count_covered_examples(matrix, vocabulary_size):
  """Returns the number of examples whose added phrases are in the vocabulary.

  This assumes the vocabulary is created simply by selecting the
  `vocabulary_size` most frequent phrases.

  Args:
    matrix: Phrase occurrence matrix with the most frequent phrases on the
      left-most columns.
    vocabulary_size: Number of most frequent phrases to include in the
      vocabulary.
  """
  # Ignore the `vocabulary_size` most frequent (i.e. leftmost) phrases (i.e.
  # columns) and count the rows with zero added phrases.
  return (matrix[:, vocabulary_size:].sum(axis=1) == 0).sum()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('output_file')

  data_iterator = utils.yield_sources_and_targets(FLAGS.input_file,
                                                  FLAGS.input_format)
  phrase_counter, all_added_phrases, max_seq_length = _added_token_counts(
      data_iterator, FLAGS.enable_swap_tag, FLAGS.max_input_examples)
  matrix = _construct_added_phrases_matrix(all_added_phrases, phrase_counter)
  num_examples = len(all_added_phrases)

  statistics_file = FLAGS.output_file + '.log'
  with tf.io.gfile.GFile(FLAGS.output_file, 'w') as writer:
    with tf.io.gfile.GFile(statistics_file, 'w') as stats_writer:
      stats_writer.write('Idx\tFrequency\tCoverage (%)\tPhrase\n')
      writer.write('KEEP\n')
      writer.write('DELETE\n')
      if FLAGS.enable_swap_tag:
        writer.write('SWAP\n')
      for i, (phrase, count) in enumerate(
          phrase_counter.most_common(FLAGS.vocabulary_size +
                                     FLAGS.num_extra_statistics)):
        # Write tags.
        if i < FLAGS.vocabulary_size:  # TODO  为什么要限制一个ｐｈｒａｓｅ既能在ＫＥＥＰ前，也能在ＤＥＬＥＴＥ前？？　
          writer.write(f'KEEP|{phrase}\n')
          writer.write(f'DELETE|{phrase}\n')
        # Write statistics.
        coverage = 100.0 * _count_covered_examples(matrix, i + 1) / num_examples # 用前ｉ＋１个高频ｐｈｒａｓｅ能覆盖的语料的比例
        stats_writer.write(f'{i+1}\t{count}\t{coverage:.2f}\t{phrase}\n')
  logging.info(f'Wrote tags to: {FLAGS.output_file}')
  logging.info(f'Wrote coverage numbers to: {statistics_file}')
  print(curLine(), "max_seq_length=", max_seq_length)


if __name__ == '__main__':
  app.run(main)
