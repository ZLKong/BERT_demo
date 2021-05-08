from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os

#from vgg import VGG
#from resnet_ma import ResNet18, ResNet50
#from testers_geng import *
from failure import *
import xlsxwriter

""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


# choose GPU or CPU
#args.cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)

#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#use_cuda = not args.no_cuda and torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

#parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
#                    help='input batch size for testing (default: 256)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='disables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

# Set seed
set_seed(training_args.seed)

try:
    num_labels = glue_tasks_num_labels[data_args.task_name]
    output_mode = glue_output_modes[data_args.task_name]
except KeyError:
    raise ValueError("Task not found: %s" % (data_args.task_name))

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

# Get datasets
train_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
)
eval_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    if training_args.do_eval
    else None
)
test_dataset = (
    GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
    if training_args.do_predict
    else None
)

def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=build_compute_metrics_fn(data_args.task_name),
)

#print(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)

# Training
if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

# Evaluation

eval_results = {}
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        eval_datasets.append(
            GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        )

    for eval_dataset in eval_datasets:
        trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
        )
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)
    #for DAC geng
    #print('eval_results!!!!!!!!!',eval_results['eval_acc'])

####################################################
# modify this part

include_layers = [
    "bert.embeddings.word_embeddings.weight", 
    "bert.embeddings.position_embeddings.weight", 
    "bert.embeddings.token_type_embeddings.weight", 
    "bert.encoder.layer.0.attention.self.query.weight", 
    "bert.encoder.layer.0.attention.self.key.weight", 
    "bert.encoder.layer.0.attention.self.value.weight", 
    "bert.encoder.layer.0.attention.output.dense.weight", 
    "bert.encoder.layer.0.intermediate.dense.weight", 
    "bert.encoder.layer.0.output.dense.weight", 
    "bert.encoder.layer.1.attention.self.query.weight", 
    "bert.encoder.layer.1.attention.self.key.weight", 
    "bert.encoder.layer.1.attention.self.value.weight", 
    "bert.encoder.layer.1.attention.output.dense.weight", 
    "bert.encoder.layer.1.intermediate.dense.weight", 
    "bert.encoder.layer.1.output.dense.weight", 
    "bert.encoder.layer.2.attention.self.query.weight", 
    "bert.encoder.layer.2.attention.self.key.weight", 
    "bert.encoder.layer.2.attention.self.value.weight", 
    "bert.encoder.layer.2.attention.output.dense.weight", 
    "bert.encoder.layer.2.intermediate.dense.weight", 
    "bert.encoder.layer.2.output.dense.weight", 
    "bert.encoder.layer.3.attention.self.query.weight", 
    "bert.encoder.layer.3.attention.self.key.weight", 
    "bert.encoder.layer.3.attention.self.value.weight", 
    "bert.encoder.layer.3.attention.output.dense.weight", 
    "bert.encoder.layer.3.intermediate.dense.weight", 
    "bert.encoder.layer.3.output.dense.weight", 
    "bert.encoder.layer.4.attention.self.query.weight", 
    "bert.encoder.layer.4.attention.self.key.weight", 
    "bert.encoder.layer.4.attention.self.value.weight", 
    "bert.encoder.layer.4.attention.output.dense.weight", 
    "bert.encoder.layer.4.intermediate.dense.weight", 
    "bert.encoder.layer.4.output.dense.weight", 
    "bert.encoder.layer.5.attention.self.query.weight", 
    "bert.encoder.layer.5.attention.self.key.weight", 
    "bert.encoder.layer.5.attention.self.value.weight", 
    "bert.encoder.layer.5.attention.output.dense.weight", 
    "bert.encoder.layer.5.intermediate.dense.weight", 
    "bert.encoder.layer.5.output.dense.weight", 
    "bert.encoder.layer.6.attention.self.query.weight", 
    "bert.encoder.layer.6.attention.self.key.weight", 
    "bert.encoder.layer.6.attention.self.value.weight", 
    "bert.encoder.layer.6.attention.output.dense.weight", 
    "bert.encoder.layer.6.intermediate.dense.weight", 
    "bert.encoder.layer.6.output.dense.weight", 
    "bert.encoder.layer.7.attention.self.query.weight", 
    "bert.encoder.layer.7.attention.self.key.weight", 
    "bert.encoder.layer.7.attention.self.value.weight", 
    "bert.encoder.layer.7.attention.output.dense.weight", 
    "bert.encoder.layer.7.intermediate.dense.weight", 
    "bert.encoder.layer.7.output.dense.weight", 
    "bert.encoder.layer.8.attention.self.query.weight", 
    "bert.encoder.layer.8.attention.self.key.weight", 
    "bert.encoder.layer.8.attention.self.value.weight", 
    "bert.encoder.layer.8.attention.output.dense.weight", 
    "bert.encoder.layer.8.intermediate.dense.weight", 
    "bert.encoder.layer.8.output.dense.weight", 
    "bert.encoder.layer.9.attention.self.query.weight", 
    "bert.encoder.layer.9.attention.self.key.weight", 
    "bert.encoder.layer.9.attention.self.value.weight", 
    "bert.encoder.layer.9.attention.output.dense.weight", 
    "bert.encoder.layer.9.intermediate.dense.weight", 
    "bert.encoder.layer.9.output.dense.weight", 
    "bert.encoder.layer.10.attention.self.query.weight", 
    "bert.encoder.layer.10.attention.self.key.weight", 
    "bert.encoder.layer.10.attention.self.value.weight", 
    "bert.encoder.layer.10.attention.output.dense.weight", 
    "bert.encoder.layer.10.intermediate.dense.weight", 
    "bert.encoder.layer.10.output.dense.weight", 
    "bert.encoder.layer.11.attention.self.query.weight", 
    "bert.encoder.layer.11.attention.self.key.weight", 
    "bert.encoder.layer.11.attention.self.value.weight", 
    "bert.encoder.layer.11.attention.output.dense.weight", 
    "bert.encoder.layer.11.intermediate.dense.weight", 
    "bert.encoder.layer.11.output.dense.weight", 
    "bert.pooler.dense.weight", 
    "classifier.weight" 
]

remove_module = False   # remove "module." in layer name

model_name = model_args.model_name_or_path+"/pytorch_model.bin"

# failure_rate_list = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
#failure_rate_list = [0.00001,0.00005, 0.0001, 0.001, 0.005]
failure_rate_list = [0.00001,0.00005,0.0001, 0.001, 0.005 ]
num_runs = 5

####################################################


#def run_failure_test(model_name, include_layers, test_name, mapping_scheme, remove_module, num_runs=20):
#test_name="double_xbar_failure"
#mapping_scheme="double"

test_name="differential_xbar_failure"
mapping_scheme="differential"

file_name = model_args.model_name_or_path.split("/")[-1]
print(file_name)
xlse_file_name = "./failure_xlsx/failure_noFirstLayer_" + test_name + file_name + ".xlsx"
workbook = xlsxwriter.Workbook(xlse_file_name)
print("open and write to file:", xlse_file_name)

worksheet = workbook.add_worksheet(test_name)

ori_acc_list = []
avg_acc_list = []
delta_acc_list = []
max_delta_acc_list = []
min_delta_acc_list = []
ori_accu = eval_results['eval_acc']  # modify here !!!!!!!!!!!!!
#ori_accu = eval_results['eval_mnli-mm/acc']
print('ori_accu',ori_accu)

for rate in failure_rate_list:

    total_failure_rate = rate
    sa0_rate = 1.75 / (1.75 + 9.04)
    sa1_rate = 9.04 / (1.75 + 9.04)
    prob_sa0 = total_failure_rate * sa0_rate
    prob_sa1 = total_failure_rate * sa1_rate
    print("prob_sa0: ", prob_sa0)
    print("prob_sa1: ", prob_sa1)

    ###############################################

    acc_list = []
    for i in range(num_runs):
        if mapping_scheme == "offset":
            make_offset_crossbar_failure(model, model_name, prob_sa0=prob_sa0, prob_sa1=prob_sa1,
                                     include_layers=include_layers,
                                     remove_module=remove_module)
        elif mapping_scheme == "double":
            make_two_normal_failure(model, model_name, prob_sa0=prob_sa0, prob_sa1=prob_sa1,
                                    include_layers=include_layers, remove_module=remove_module)
        elif mapping_scheme == "differential":
            make_two_differential_crossbar_failure(model, model_name, prob_sa0=prob_sa0, prob_sa1=prob_sa1,
                                                   include_layers=include_layers, remove_module=remove_module)

        print("run:", i)

        ###############################################
        eval_results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            eval_datasets = [eval_dataset]
            if data_args.task_name == "mnli":
                mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
                eval_datasets.append(
                    GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
                )

            for eval_dataset in eval_datasets:
                trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                output_eval_file = os.path.join(
                    training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
                )
                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

                eval_results.update(eval_result)
            #for DAC geng
            #print('eval_results!!!!!!!!!',eval_results['eval_acc'])
            accu =eval_results['eval_acc']
            #accu =eval_results['eval_mnli-mm/acc']

        ###############################################

        acc_list.append(accu)

    avg_acc = sum(acc_list) / num_runs
    delta_acc = ori_accu - avg_acc
    max_delta_acc = ori_accu - min(acc_list)
    min_delta_acc = ori_accu - max(acc_list)
    print("acc_list:", acc_list)
    print("average acc:", avg_acc)
    print("delta acc:", delta_acc)
    print("max_delta_acc", max_delta_acc)
    print("min_delta_acc", min_delta_acc)

    ori_acc_list.append(ori_accu)
    avg_acc_list.append(avg_acc)
    delta_acc_list.append(delta_acc)
    max_delta_acc_list.append(max_delta_acc)
    min_delta_acc_list.append(min_delta_acc)

print("writing rate:{} model:{} results to excel ...".format(rate, model_name))

start_col = 1

worksheet.write_row(1, start_col, failure_rate_list)
worksheet.write_row(2, start_col, ori_acc_list)
worksheet.write_row(3, start_col, avg_acc_list)
worksheet.write_row(4, start_col, delta_acc_list)
worksheet.write_row(5, start_col, max_delta_acc_list)
worksheet.write_row(6, start_col, min_delta_acc_list)

workbook.close()
print("writing finished!")



