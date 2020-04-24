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

"""Classes representing a tag and a text editing task.

Tag corresponds to an edit operation, while EditingTask is a container for the
input that LaserTagger takes. EditingTask also has a method for realizing the
output text given the predicted tags.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from enum import Enum
from curLine_file import curLine

import utils


class TagType(Enum):
  """Base tag which indicates the type of an edit operation."""
  # Keep the tagged token.
  KEEP = 1
  # Delete the tagged token.
  DELETE = 2
  # Keep the tagged token but swap the order of sentences. This tag is only
  # applied if there are two source texts and the tag is applied to the last
  # token of the first source. In other contexts, it's treated as KEEP.
  SWAP = 3


class Tag(object):
  """Tag that corresponds to a token edit operation.

  Attributes:
    tag_type: TagType of the tag.
    added_phrase: A phrase that's inserted before the tagged token (can be
      empty).
  """

  def __init__(self, tag):
    """Constructs a Tag object by parsing tag to tag_type and added_phrase.

    Args:
      tag: String representation for the tag which should have the following
        format "<TagType>|<added_phrase>" or simply "<TagType>" if no phrase
        is added before the tagged token. Examples of valid tags include "KEEP",
        "DELETE|and", and "SWAP|.".

    Raises:
      ValueError: If <TagType> is invalid.
    """
    if '|' in tag:
      pos_pipe = tag.index('|') # 也可以直接split，再把［１：］拼接成一个字符串吧
      tag_type, added_phrase = tag[:pos_pipe], tag[pos_pipe + 1:]
    else:
      tag_type, added_phrase = tag, ''
    try:
      self.tag_type = TagType[tag_type] # for example: tag_type:KEEP self.tag_type:TagType.KEEP
    except KeyError:
      raise ValueError(
          'TagType should be KEEP, DELETE or SWAP, not {}'.format(tag_type))
    self.added_phrase = added_phrase

  def __str__(self):
    if not self.added_phrase:
      return self.tag_type.name
    else:
      return '{}|{}'.format(self.tag_type.name, self.added_phrase)


class EditingTask(object):
  """Text-editing task.

  Attributes:
    sources: Source texts.
    source_tokens: Tokens of the source texts concatenated into a single list.
    first_tokens: The indices of the first tokens of each source text.
  """

  def __init__(self, sources, location=None):
    """Initializes an instance of EditingTask.

    Args:
      sources: A list of source strings. Typically contains only one string but
        for sentence fusion it contains two strings to be fused (whose order may
        be swapped).
      location: None或字符串, 0表示能变,1表示不能变
    """
    self.sep='' # for Chinses
    self.sources = sources
    source_token_lists = [utils.get_token_list(text) for text in self.sources]
    # Tokens of the source texts concatenated into a single list.
    self.source_tokens = []
    # The indices of the first tokens of each source text.
    self.first_tokens = []
    for token_list in source_token_lists:
      self.first_tokens.append(len(self.source_tokens))
      self.source_tokens.extend(token_list)
    self.location = location

  def _realize_sequence(self, tokens, tags):
    """Realizes output text corresponding to a single source text.

    Args:
      tokens: Tokens of the source text.
      tags: Tags indicating the edit operations.

    Returns:
      The realized text.
    """
    output_tokens = []
    # add_flag = True # 可以插入
    for index, (token, tag) in enumerate(zip(tokens, tags)):
      loc = "0"
      if self.location is not None:
        loc = self.location[index]
      if tag.added_phrase and (loc == "0" or index==0 or (index>0 and self.location[index-1]=="0")):  # TODO
        output_tokens.append(tag.added_phrase)
      if tag.tag_type in (TagType.KEEP, TagType.SWAP) or loc == "1":  # TODO 根据需要修改代码,location为"1"的位置不能被删除, 但目前是可以插入的
        output_tokens.append(token)
    return self.sep.join(output_tokens)

  def _first_char_to_upper(self, text):
    """Upcases the first character of the text."""
    try:
      return text[0].upper() + text[1:]
    except IndexError:
      return text

  def _first_char_to_lower(self, text):
    """Lowcases the first character of the text."""
    try:
      return text[0].lower() + text[1:]
    except IndexError:
      return text

  def realize_output(self, tags):
    """Realize output text based on the source tokens and predicted tags.

    Args:
      tags: Predicted tags (one for each token in `self.source_tokens`).

    Returns:
      The realizer output text.

    Raises:
      ValueError: If the number of tags doesn't match the number of source
        tokens.
    """
    if len(tags) != len(self.source_tokens):
      raise ValueError('The number of tags ({}) should match the number of '
                       'source tokens ({})'.format(
                           len(tags), len(self.source_tokens)))
    outputs = []  # Realized sources that are joined into the output text.
    if (len(self.first_tokens) == 2 and
        tags[self.first_tokens[1] - 1].tag_type == TagType.SWAP):
      order = [1, 0]
    else:
      order = range(len(self.first_tokens))
    for source_idx in order:
      # Get the span of tokens for the source: [first_token, last_token).
      first_token = self.first_tokens[source_idx]
      if source_idx + 1 < len(self.first_tokens):
        last_token = self.first_tokens[source_idx + 1]  # Not inclusive.
      else:
        last_token = len(self.source_tokens)
      # Realize the source and fix casing.
      realized_source = self._realize_sequence(
          self.source_tokens[first_token:last_token],
          tags[first_token:last_token])

      if outputs:
        if len(outputs[0][-1:])>0 and outputs[0][-1:] in '.!?':
          realized_source = self._first_char_to_upper(realized_source)  # 变大写
        else:
          # Note that ideally we should also test here whether the first word is
          # a proper noun or an abbreviation that should always be capitalized.
          realized_source = self._first_char_to_lower(realized_source)  # 变小写
        # print(curLine(), len(outputs[0][-1:]), "outputs[0][-1:]:", outputs[0][-1:], "source_idx=",source_idx, ",realized_source:", realized_source)
      outputs.append(realized_source)
    prediction = self.sep.join(outputs)
    return prediction
