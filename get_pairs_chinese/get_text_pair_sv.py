# coding=utf-8
# 从SV导出后的xlsx文件，相同含义（至少是答案相同）的句子在一行，从中采样得到句子对(A,B),然后训练模型把Ａ改写成Ｂ
# 当前是针对宝安机场　
import sys
sys.path.append("..")
import os
import xlrd  # 引入模块
import random
from compute_lcs import _compute_lcs
from curLine_file import curLine

def process(corpus_folder, raw_file_name):
  raw_file = os.path.join(corpus_folder, raw_file_name)
  # 打开文件，获取excel文件的workbook（工作簿）对象
  workbook = xlrd.open_workbook(raw_file)  # 文件路径

  # 通过sheet索引获得sheet对象
  worksheet = workbook.sheet_by_index(0)
  nrows = worksheet.nrows  # 获取该表总行数
  ncols = worksheet.ncols  # 获取该表总列数
  print(curLine(), "raw_file_name:%s, worksheet:%s nrows=%d, ncols=%d" % (raw_file_name, worksheet.name,nrows, ncols))
  assert ncols == 3
  assert nrows > 0
  col_data = worksheet.col_values(0)  # 获取第一列的内容
  corpus_list = []
  for line in col_data:
    sent_list = line.strip().split("&&")
    sent_num = len(sent_list)
    for i in range(1, sent_num, 2):
      source= sent_list[i-1]
      target = sent_list[i]
      # source_length = len(source)
      # if source_length > 8 and (i+1)%4>0: # 对５０％的长句随机删除
      #   rand = random.uniform(0.1, 0.9)
      #   source_pre = source
      #   swag_location = int(source_length*rand)
      #   source = "%s%s" % (source[:swag_location], source[swag_location+1:])
      #   print(curLine(), "source_pre:%s, source:%s" % (source_pre, source))

      length = float(len(source) + len(target))
      lcs1 = _compute_lcs(source, target)
      lcs_rate= len(lcs1)/length
      if (lcs_rate<0.2):
        continue # 变动过大，忽略

      # if (lcs_rate<0.3):
      #   continue # 变动过大，忽略
      # if len(source)*1.15 < len(target):
      #   new_t = source
      #   source = target
      #   target = new_t
      corpus = "%s\t%s\t%f\n" % (source, target, lcs_rate)
      corpus_list.append(corpus)
  return corpus_list

def main(corpus_folder, save_folder):
  fileList = os.listdir(corpus_folder)
  corpus_list_total = []
  for raw_file_name in fileList:
    corpus_list = process(corpus_folder, raw_file_name)
    print(curLine(), raw_file_name, len(corpus_list))
    corpus_list_total.extend(corpus_list)
  save_file = os.path.join(save_folder, "baoan_airport_from_xlsx.txt")
  with open(save_file, "w") as fw:
    fw.writelines(corpus_list_total)
  print(curLine(), "have save %d to %s" % (len(corpus_list_total), save_file))




if __name__ == "__main__":
  corpus_folder = "/home/cloudminds/Mywork/corpus/Chinese_QA/baoanairport/agent842-3月2日"

  save_folder = "/home/cloudminds/Mywork/corpus/rephrase_corpus"
  raw_file_name = "专业知识导出记录.xlsx"
  main(corpus_folder, save_folder)

