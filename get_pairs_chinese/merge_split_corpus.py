# coding=utf-8
# 对不同来源的语料融合，然后再划分出ｔｒａｉｎ，ｄｅｖ，ｔｅｓｔ
import os
import numpy as np
from curLine_file import curLine

def merge(raw_file_name_list, save_folder):
  corpus_list = []
  for raw_file_name in raw_file_name_list:
    raw_file = os.path.join(save_folder, "%s.txt" % raw_file_name)
    with open(raw_file) as fr:
      lines = fr.readlines()
    corpus_list.extend(lines)
    if "baoan" in raw_file_name:
      corpus_list.extend(lines) # TODO
  return corpus_list

def split(corpus_list, save_folder, trainRate=0.8):
  corpusNum = len(corpus_list)
  shuffle_indices = list(np.random.permutation(range(corpusNum)))
  indexTrain = int(trainRate * corpusNum)
  # indexDev= int((trainRate + devRate) * corpusNum)
  corpusTrain = []
  for i in shuffle_indices[:indexTrain]:
    corpusTrain.append(corpus_list[i])
  save_file = os.path.join(save_folder, "train.txt")
  with open(save_file, "w") as fw:
    fw.writelines(corpusTrain)
  print(curLine(), "have save %d to %s" % (len(corpusTrain), save_file))

  corpusDev = []
  for i in shuffle_indices[indexTrain:]:  # TODO all corpus
    corpusDev.append(corpus_list[i])
  save_file = os.path.join(save_folder, "tune.txt")
  with open(save_file, "w") as fw:
    fw.writelines(corpusDev)
  print(curLine(), "have save %d to %s" % (len(corpusDev), save_file))


  save_file = os.path.join(save_folder, "test.txt")
  with open(save_file, "w") as fw:
    fw.writelines(corpusDev)
  print(curLine(), "have save %d to %s" % (len(corpusDev), save_file))




if __name__ == "__main__":
  raw_file_name = ["baoan_airport", "lcqmc", "baoan_airport_from_xlsx"]
  save_folder = "/home/cloudminds/Mywork/corpus/rephrase_corpus"
  corpus_list = merge(raw_file_name, save_folder)
  split(corpus_list, save_folder, trainRate=0.8)

