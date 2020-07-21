import logging
import math
import os
import random
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, default_data_collator
from .file_utils import is_apex_available, is_torch_tpu_available
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput, is_wandb_available
from .training_args import TrainingArguments
import yaml
import json
import prune_util

import testers

if is_apex_available():
    from apex import amp


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        self.data_collator = data_collator if data_collator is not None else default_data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            if self.args.rew:
                self.tb_writer = SummaryWriter(str(self.args.logging_dir)+'/reweighted_l1_'+str(self.args.penalty_config_file)+'_lr_'+str(self.args.learning_rate)+'_lr_decay_'+str(self.args.lr_decay))
                self.log_name = "reweighted_training"
                self.output_dir_saved_model = self.log_name+'_'+str(self.args.logging_dir.split("/")[1])+'_'+str(self.args.penalty_config_file)+'_lr_'+str(self.args.learning_rate)+'_lr_decay_'+str(self.args.lr_decay)
            elif self.args.masked_retrain:
                if self.args.prune_ratio_config != None:
                    self.tb_writer = SummaryWriter(str(self.args.logging_dir)+'/reweighted_l1_'+'lr_retrain_'+str(self.args.lr_retrain)+'_lr_decay_'+str(self.args.lr_decay)+'_'+str(self.args.prune_ratio_config))
                    self.log_name = "reweighted_retraining"
                    self.output_dir_saved_model = self.log_name+'_'+str(self.args.logging_dir.split("/")[1])+'_lr_retrain_'+str(self.args.lr_retrain)+'_lr_decay_'+str(self.args.lr_decay)+'_'+str(self.args.prune_ratio_config)
                else:
                    self.tb_writer = SummaryWriter(str(self.args.logging_dir)+'/reweighted_l1_'+'lr_retrain_'+str(self.args.lr_retrain)+'_lr_decay_'+str(self.args.lr_decay)+'_'+str(self.args.prune_config_file))      
                    self.log_name = "reweighted_retraining"
                    self.output_dir_saved_model = self.log_name+'_'+str(self.args.logging_dir.split("/")[1])+'_lr_retrain_'+str(self.args.lr_retrain)+'_lr_decay_'+str(self.args.lr_decay)+'_'+str(self.args.prune_config_file)
            else:
                self.tb_writer = SummaryWriter(str(self.args.logging_dir)+'/huggingface_training'+'_lr_'+str(self.args.learning_rate)+'_lr_decay_'+str(self.args.lr_decay))
                self.log_name = "reweighted_training"
                self.output_dir_saved_model = "Huggingface_"+str(self.args.logging_dir.split("/")[1])+'_lr_'+str(self.args.learning_rate)+'_lr_decay_'+str(self.args.lr_decay)

      
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                    "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                    + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        if self.args.rew:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        elif self.args.masked_retrain:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr_retrain, eps=self.args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        if self.is_world_master():
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(
                    self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )


        if self.args.rew:
            # penalty:
            penalty_factors = []
            with open("./profile/" + self.args.penalty_config_file + ".yaml", "r") as stream:
                try:
                    raw_dict = yaml.load(stream)
                    penalty_factors = raw_dict['penalty_factors']
                except yaml.YAMLError as exc:
                    print(exc)


            # Display names and weights of the model.named_parameters():
            layers = []
            layers_names = []
            iii = 0
            for name, param in (model.named_parameters()):
                if len(param.shape) >= 2:
                    iii += 1
                    layers.append(param)
                    layers_names.append(name)
                    # print(name, "\n", list(param.shape))
                    # print(iii,'th parameter:''\tlen:', len(param.shape), '\t\t', name, '\t\tWeight shape: ', param.shape)
                    # print('    {}:\n        {}'.format(name,str(0.001)))
                    #print(name)
            print('len(layers) = ', len(layers))


            # initialize rew_layer
            eps = 1e-3
            block_row_division = self.args.block_row_division
            block_row_width = self.args.block_row_width
            rew_layers = []
            for i in range(len(layers)):
                conv_layer = layers[i]

                if (self.args.sparsity_type == "block_filter"): # -libn
                    shape = conv_layer.shape
                    conv = conv_layer.reshape(shape[0],-1)

                    print('conv.shape',conv.shape)
                    print('conv.shape[0]',conv.shape[0])
                    print('len(conv)',len(conv))
                    print('conv.sum(0)[0]',conv.sum(0)[0])
                    conv_mem = torch.chunk(conv, 16 , dim=0)
                    for j in range(len(conv_mem)):
                        conv_curr=conv_mem[j]
                        print('convfrag[0].shape',conv_curr.shape)
                        print('convfrag[0].sum(0)',conv_curr.sum(0))
                        for i in range(conv_curr.shape[1]): #column number 
                            print('conv_curr.sum(0)[i]',conv_curr.sum(0)[i])
                            if conv_curr.sum(0)[i] >= 0:
                                print('positive')
                                for q in range(len(conv_curr)):  #row number
                                    conv_curr[q][i] = torch.max(conv_curr[q][i].cpu(),torch.tensor([0.]))


                           # print('new mat.sum(0)[i]',conv.sum(0)[i])
                            else:
                                print('negative')
                                for q in range(len(conv_curr)):
                                    conv_curr[q][i] = torch.min(conv_curr[q][i].cpu(),torch.tensor([0.]))

                            print('new conv_curr.sum(0)[i]',conv_curr.sum(0)[i])



                    new_conv = None
                    for n in range(len(conv_mem)):
                        if new_conv is None:
                            new_conv = conv_mem[j]
                        else:
                            new_conv = torch.cat((new_conv,conv_mem[j]),0)



                    conv=new_conv
                    print('conv=new_conv',conv.shape)

                    if self.args.block_row_division != 0:
                        print("in loop",conv.shape[0], conv.shape[1])
                       # print('conv.shape[1]%block_row_width',conv.shape[1]%block_row_width)
                        if conv.shape[1]%self.args.block_row_division != 0 :
                            print('conv.shape[1]%block_row_division',conv.shape[1]%self.args.block_row_division)
                            print("the layer size is not divisible:",conv.shape[0], conv.shape[1])
                            # raise SyntaxError("block_size error")
                        block_row_division = int(conv.shape[1]/self.args.block_row_division)
                        print('change block_row_division',block_row_division)
                    else:
                        if conv.shape[1]%self.args.block_row_division != 0 :
                            print("the layer size is not divisible by block_row_division",conv.shape[1], block_row_division)
                            # raise SyntaxError("block_size error")
                    convfrag = torch.chunk(conv, block_row_division, dim=1)

                    mat = None
                    for j in range(len(convfrag)):
                        if mat is None:
                            mat = convfrag[j]
                        else:
                            mat = torch.cat((mat,convfrag[j]),0)

                    rew_layers.append(1 / (torch.norm(mat.data, dim=1) + eps))

                elif self.args.sparsity_type == "block_column":
                    shape = conv_layer.shape
                    conv = conv_layer.reshape(shape[0],-1)
                    print('weight.shape', conv.shape)  #([30522, 768])
                    if shape[0] == 30522:
                        conv = conv.expand(30720, 768)
                    if conv.shape[0]%block_row_width != 0 :
                        print("the layer size is not divisible",conv.shape[0], block_row_width)
                        # raise SyntaxError("block_size error")
                    convfrag = torch.chunk(conv, block_row_width, dim=0)

                    mat = None
                    for j in range(len(convfrag)):
                        if mat is None:
                            mat = convfrag[j]
                        else:
                            mat = torch.cat((mat,convfrag[j]),1)

                    rew_layers.append(1 / (torch.norm(mat.data, dim=0) + eps))
                else:
                    raise SyntaxError("Unknown sparsity type")

            milestone = [4,8,12,16]


        # reweighted retraining: hard_prune + normal training. -libn
        if self.args.masked_retrain:
            prune_thresholds = []
            if self.args.prune_ratio_config != None:
                prune_file_name = self.args.prune_ratio_config
                with open("./profile/" + self.args.prune_ratio_config + ".yaml", "r") as stream:
                    try:
                        raw_dict = yaml.load(stream)
                        prune_thresholds = raw_dict['prune_ratios']
                    except yaml.YAMLError as exc:
                        print(exc)
            else:
                prune_file_name = self.args.prune_config_file
                with open("./profile/" + self.args.prune_config_file + ".yaml", "r") as stream:
                    try:
                        raw_dict = yaml.load(stream)
                        prune_thresholds = raw_dict['prune_thresholds']
                    except yaml.YAMLError as exc:
                        print(exc)

            prune_util.hard_prune(self.args, prune_thresholds, model)


        l1_loss = 0

        for epoch in train_iterator:

            if self.args.rew:
                # training mask: to ensure that pruned weights keep being pruned. -libn
                print("\nprogressive admm-train/re-train masking")
                masks = {}
                for name, W in (model.named_parameters()):
                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).to(self.args.device)
                    W = torch.from_numpy(weight).to(self.args.device)
                    W.data = W
                    masks[name] = zero_mask

            elif self.args.masked_retrain:
                print("\nfull acc re-train masking")
                masks = {}
                for name, W in (model.named_parameters()):
                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).to(self.args.device)
                    W = torch.from_numpy(weight).to(self.args.device)
                    W.data = W
                    masks[name] = zero_mask




            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
           # print("!!!!!!!!!!!!!!!!! len(epoch_iterator) = ", len(epoch_iterator))
            for step, inputs in enumerate(epoch_iterator):

                # print("!!!!!!!!!!!!!!!!! len(epoch_iterator) = ", len(epoch_iterator))

                if self.args.rew:
                    l1_loss = 0
                    # calculate R of each layer according to milestone. -libn
                    if step == 0 and epoch in milestone:
                        print("reweighted l1 update")
                        for j in range(len(layers)):

                            if (self.args.sparsity_type == "block_filter"): # -libn
                                shape = layers[j].shape
                                conv = layers[j].reshape(shape[0], -1)      

                                if block_row_width != 0:
                                    if conv.shape[1]%block_row_width != 0 :
                                        print("the layer size is not divisible by block_row_width:",conv.shape[1], block_row_width)
                                        # raise SyntaxError("block_size error")
                                    block_row_division = int(conv.shape[1]/block_row_width)
                                else:
                                    if conv.shape[1]%block_row_division != 0 :
                                        print("the layer size is not divisible by block_row_division",conv.shape[1], block_row_division)
                                        # raise SyntaxError("block_size error")
                                convfrag = torch.chunk(conv, block_row_division, dim=1)

                                mat = None
                                for k in range(len(convfrag)):
                                    if mat is None:
                                        mat = convfrag[k]                       # if block_row_division = 8, convfrag[j].shape=[64,4]. -libn
                                    else:
                                        mat = torch.cat((mat, convfrag[k]), 0)  # mat.shape=[64*num_blocks, 4]. -libn
                                rew_layers[j] = (1 / (torch.norm(mat.data, dim=1) + eps))   # calculate the l2 norm of each row of mat. -> rew_layers[j].shape=[64*num_blocks]. -libn

                            elif self.args.sparsity_type == "block_column":
                                shape = layers[j].shape
                                conv = layers[j].reshape(shape[0], -1)      # conv.shape=[64, 27]. -libn 
                                # print('weight.shape', conv.shape)   
                                if shape[0] == 30522:
                                    conv = conv.expand(30720, 768)
                                if conv.shape[0] % block_row_width != 0:
                                    print("the layer size (cross_f) is not divisible", conv.shape[0], block_row_width)
                                    # raise SyntaxError("block_size error")
                                convfrag = torch.chunk(conv, block_row_width, dim=0)   # if cross_f=8, convfrag[j].shape=[8,27]. -libn

                                mat = None
                                for k in range(len(convfrag)):
                                    if mat is None:
                                        mat = convfrag[k]
                                    else:
                                        mat = torch.cat((mat, convfrag[k]), 1)
                                rew_layers[j] = (1 / (torch.norm(mat.data, dim=0) + eps))   
                            else:
                                raise SyntaxError("Unknown sparsity type")

                    # calculate l1_loss of each layer every epoch. -libn
                    for j in range(len(layers)):
                        rew = rew_layers[j]
                        conv_layer = layers[j]

                        # block-filter:
                        if (self.args.sparsity_type == "block_filter"): # -libn
                            shape = layers[j].shape
                            conv = layers[j].reshape(shape[0], -1)

                            if block_row_width != 0:
                                if conv.shape[1]%block_row_width != 0 :
                                    print("the layer size is not divisible by block_row_width:",conv.shape[1], block_row_width)
                                    # raise SyntaxError("block_size error")
                                block_row_division = int(conv.shape[1]/block_row_width)
                            else:
                                if conv.shape[1]%block_row_division != 0 :
                                    print("the layer size is not divisible by block_row_division",conv.shape[1], block_row_division)
                                    # raise SyntaxError("block_size error")
                            convfrag = torch.chunk(conv, block_row_division, dim=1)

                            mat = None
                            for k in range(len(convfrag)):
                                if mat is None:
                                    mat = convfrag[k]
                                else:
                                    mat = torch.cat((mat, convfrag[k]), 0)
                            l1_loss = l1_loss + penalty_factors[layers_names[j]] * torch.sum(rew * torch.norm(mat, dim=1))

                        elif self.args.sparsity_type == "block_column":
                            shape = layers[j].shape
                            conv = layers[j].reshape(shape[0], -1)
                            # print('weight.shape', conv.shape)
                            if shape[0] == 30522:
                                conv = conv.expand(30720, 768)
                            if conv.shape[0] % block_row_width != 0:
                                print("the layer size (cross_f) is not divisible", conv.shape[0], block_row_width)
                                # raise SyntaxError("block_size error")

                            convfrag = torch.chunk(conv, block_row_width, dim=0)

                            mat = None
                            for k in range(len(convfrag)):
                                if mat is None:
                                    mat = convfrag[k]
                                else:
                                    mat = torch.cat((mat, convfrag[k]), 1)

                            l1_loss = l1_loss + penalty_factors[layers_names[j]] * torch.sum(rew * torch.norm(mat, dim=0))

                        else:
                            raise SyntaxError("Unknown sparsity type")
                        
                        
                        # print("!!! Layer name: ", layers_names[j], "penalty_factor: ", penalty_factors[layers_names[j]])
                
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(l1_loss, model, inputs, optimizer)

                # mask the gradient for retraining: -libn
                if self.args.masked_retrain:
                    # with torch.no_grad():
                    for name, W in (model.named_parameters()):
                        # print("model:",self.args.logging_dir.split("/")[0][-5:])
                        # print("!!!!!",name)
                        if "mask_emb" in name:
                            continue
                        if name in masks:
                            W.grad *= masks[name]
                            # print("!!!!!! Works well!")


                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    # scheduler.step()
                    # default lr scheduler:
                    if self.args.lr_decay == 0.0:
                        scheduler.step()  # Update learning rate schedule
                    
                    model.zero_grad()
                    self.global_step += 1

                    if self.args.local_rank in [-1, 0]:
                        if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            if self.args.evaluate_during_training:
                                results = self.evaluate()
                                for key, value in results.items():
                                    eval_key = "eval_{}".format(key)
                                    logs[eval_key] = value

                            loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                            learning_rate_scalar = scheduler.get_last_lr()[0]
                            logs["learning_rate"] = learning_rate_scalar
                            
                            if self.args.rew:
                                logs["mixed_loss"] = loss_scalar
                            else:
                                logs["loss"] = loss_scalar
                            logging_loss = tr_loss

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(self.args.logging_dir.split("/")[1]+'/'+self.log_name+'/'+k, v, self.global_step)
                            epoch_iterator.write(json.dumps({**logs, **{"step": self.global_step}}))

                        if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert model.module is self.model
                            else:
                                assert model is self.model
                            # Save model checkpoint
                            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                            self.save_model(output_dir)
                            self._rotate_checkpoints()
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            # # update lr each epoch: -libn
            # if self.args.lr_decay != 0.0:    # customized lr scheduler:
            #     for param_group in optimizer.param_groups:
            #         current_lr = param_group['lr']
            #     self.tb_writer.add_scalar(self.args.logging_dir.split("/")[1]+'/'+self.log_name+'/lr', current_lr, int(epoch))
            # else:                       # default lr scheduler:
            #     self.tb_writer.add_scalar(self.args.logging_dir.split("/")[1]+'/'+self.log_name+'/lr', scheduler.get_last_lr()[0], int(epoch))

            if self.args.masked_retrain:
                compression_rate = testers.test_irregular_sparsity(model)
                self.tb_writer.add_scalar(self.args.logging_dir.split("/")[1]+'/'+self.log_name+'/compression_rate', compression_rate, self.epoch)  # Save the calculated compression rate only once! -libn
    

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_master():
                wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)

    def _training_step(
        self, l1_loss: float, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.rew:
            loss = l1_loss + loss

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output
