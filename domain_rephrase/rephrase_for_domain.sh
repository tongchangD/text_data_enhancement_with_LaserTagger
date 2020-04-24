# 扩充技能的语料
# rephrase_for_skill.sh: 在rephrase.sh基础上改的
# predict_for_skill.py: 在 predict_main.py基础上改的
# score_for_skill.txt 结果对比


#　成都
# pyenv activate python373tf115
# pip install -i https://pypi.douban.com/simple/ bert-tensorflow==1.0.1
#pip install -i https://pypi.douban.com/simple/ tensorflow==1.15.0
#python -m pip install --upgrade pip -i https://pypi.douban.com/simple

# set gpu id to use
export CUDA_VISIBLE_DEVICES=""

# 房山
# pyenv activate python363tf111
# pip install bert-tensorflow==1.0.1

#scp -r /home/cloudminds/PycharmProjects/lasertagger-Chinese/predict_main.py  cloudminds@10.13.33.128:/home/cloudminds/PycharmProjects/lasertagger-Chinese
#scp -r cloudminds@10.13.33.128:/home/wzk/Mywork/corpus/文本复述/output/models/wikisplit_experiment_name /home/cloudminds/Mywork/corpus/文本复述/output/models/
# watch -n 1 nvidia-smi

start_tm=`date +%s%N`;

export HOST_NAME="cloudminds" #　 　"wzk" #
### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=wikisplit_experiment
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
NUM_EPOCHS=10.0
export TRAIN_BATCH_SIZE=256  # 512 OOM   256 OK
PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
SAVE_CHECKPOINT_STEPS=200
export enable_swap_tag=false
export output_arbitrary_targets_for_infeasible_examples=false
export WIKISPLIT_DIR="/home/${HOST_NAME}/Mywork/corpus/rephrase_corpus"
export OUTPUT_DIR="${WIKISPLIT_DIR}/output"

#python phrase_vocabulary_optimization.py \
#  --input_file=${WIKISPLIT_DIR}/train.txt \
#  --input_format=wikisplit \
#  --vocabulary_size=500 \
#  --max_input_examples=1000000 \
#  --enable_swap_tag=${enable_swap_tag} \
#  --output_file=${OUTPUT_DIR}/label_map.txt


export max_seq_length=40 # TODO
export BERT_BASE_DIR="/home/${HOST_NAME}/Mywork/model/RoBERTa-tiny-clue" # chinese_L-12_H-768_A-12"




# Check these numbers from the "*.num_examples" files created in step 2.
export CONFIG_FILE=configs/lasertagger_config.json
export EXPERIMENT=wikisplit_experiment_name



### 4. Prediction

# Export the model.
#python run_lasertagger.py \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --model_config_file=${CONFIG_FILE} \
#  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
#  --do_export=true \
#  --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export

## Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}
PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred.tsv

python domain_rephrase/predict_for_domain.py \
  --input_file=/home/${HOST_NAME}/Mywork/corpus/ner_corpus/domain_corpus/train3.csv \
  --input_format=wikisplit \
  --output_file=/home/${HOST_NAME}/Mywork/corpus/ner_corpus/domain_corpus/train3_expand.csv \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --saved_model=${SAVED_MODEL_DIR}

#### 5. Evaluation
#python score_main.py --prediction_file=${PREDICTION_FILE}


end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"