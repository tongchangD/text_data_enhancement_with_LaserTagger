# coding=utf-8
# 世西给的ＱＡ对（至少是答案相同）的句子在一行，从中采样得到句子对(A,B),然后训练模型把Ａ改写成Ｂ
# 当前是针对宝安机场　比较旧且少的数据集，
import os
from compute_lcs import _compute_lcs
from curLine_file import curLine

def process(corpus_folder, raw_file_name, save_folder):
  raw_file = os.path.join(corpus_folder, raw_file_name)
  with open(raw_file, "r") as fr:
    lines = fr.readlines()
  corpus_list = []
  for line in lines:
    sent_list = line.strip().split("&&")
    sent_num = len(sent_list)
    for i in range(1, sent_num, 2):
      source= sent_list[i-1]
      target = sent_list[i]
      length = float(len(source) + len(target))
      lcs1 = _compute_lcs(source, target)
      lcs_rate= len(lcs1)/length
      if (lcs_rate<0.3):
        continue # 变动过大，忽略
      if len(source)*1.15 < len(target):
        new_t = source
        source = target
        target = new_t
      corpus = "%s\t%s\t%f\n" % (source, target, lcs_rate)
      corpus_list.append(corpus)
  save_file = os.path.join(save_folder, "baoan_airport.txt")
  with open(save_file, "w") as fw:
    fw.writelines(corpus_list)
  print(curLine(), "have save %d to %s" % (len(corpus_list), save_file))




if __name__ == "__main__":
  corpus_folder = "/home/cloudminds/Mywork/corpus/Chinese_QA/baoanairport"
  raw_file_name = "baoan_airport_processed.txt"
  save_folder = "/home/cloudminds/Mywork/corpus/rephrase_corpus"
  process(corpus_folder, raw_file_name, save_folder)

