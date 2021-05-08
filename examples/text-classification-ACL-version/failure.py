import torch
import numpy as np
import random
from collections import OrderedDict
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)



# def failure_sa0(weight, prob_sa0):
#
#     weight_np = weight.cpu().detach().numpy()
#     shape = weight_np.shape
#     weight1d = weight_np.reshape(-1)
#     # print(weight1d.shape)
#     num_of_weights = weight1d.shape[0]
#
#     num_of_fails = int(num_of_weights * prob_sa0)
#
#     fail_idx = []
#     for i in range(num_of_fails):
#         rand_idx = random.randint(0, num_of_weights - 1)
#         fail_idx.append(rand_idx)
#
#     weight1d[fail_idx] = 0
#     weight4d = weight1d.reshape(shape)
#     weight4d = torch.from_numpy(weight4d).cuda()
#
#     return weight4d
#
#
# def failure_sa1(weight, prob_sa1, scale=1):
#
#     weight_np = weight.cpu().detach().numpy()
#     shape = weight_np.shape
#     weight1d = weight_np.reshape(-1)
#     # print(weight1d.shape)
#     num_of_weights = weight1d.shape[0]
#
#     num_of_fails = int(num_of_weights * prob_sa1)
#
#     fail_idx = []
#     for i in range(num_of_fails):
#         rand_idx = random.randint(0, num_of_weights - 1)
#         fail_idx.append(rand_idx)
#
#     maximum_value = 1 * scale
#
#     weight1d[fail_idx] = maximum_value
#     weight4d = weight1d.reshape(shape)
#     weight4d = torch.from_numpy(weight4d).cuda()
#
#     return weight4d


def failure_to_value(weight, prob_fail=0, value=0.0):

    weight_np = weight.cpu().detach().numpy()
    shape = weight_np.shape
    weight1d = weight_np.reshape(-1)

    num_of_weights = weight1d.shape[0]

    num_of_fails = int(num_of_weights * prob_fail)

    idx_list = list(range(num_of_weights))
    fail_idx = random.sample(idx_list, num_of_fails)

    weight1d[fail_idx] = value
    weight4d = weight1d.reshape(shape)
    weight4d = torch.from_numpy(weight4d).cuda()

    return weight4d



def convert_to_two_differential_crossbar(w):
    with torch.no_grad():

        scale = torch.max(torch.abs(w))
        w = w / scale

        shape = w.shape
        w1 = torch.ones(shape, requires_grad=False).cuda()
        w2 = torch.ones(shape, requires_grad=False).cuda()

        positive_idx = (w >= 0)
        negative_idx = (w < 0)

        w2[positive_idx] = w1[positive_idx] - w[positive_idx]
        w1[negative_idx] = w2[negative_idx] + w[negative_idx]

        # change type from Tensor -> ndarray -> float
        scale = scale.cpu().detach().numpy().item()

    return scale, w1, w2

def convert_differential_to_single_crossbar(scale, w1, w2):
    """ weight.data = convert_to_single_crossbar(scale, w1, w2) """

    w = w1 - w2
    w = w * scale

    return w


def convert_to_two_normal_crossbar(w):
    with torch.no_grad():

        shape = w.shape
        w1 = torch.zeros(shape, requires_grad=False).cuda()
        w2 = torch.zeros(shape, requires_grad=False).cuda()

        positive_idx = (w > 0)
        negative_idx = (w < 0)

        w1[positive_idx] = w[positive_idx]
        w2[negative_idx] = w[negative_idx]

    return w1, w2

def convert_normal_to_single_crossbar(w1, w2):
    """ weight.data = convert_to_single_crossbar(scale, w1, w2) """

    w = w1 + w2

    return w



def normalize_weight(w):
    with torch.no_grad():
        scale = torch.max(torch.abs(w))
        w = w / scale

        # change type from Tensor -> ndarray -> float
        scale = scale.numpy().item()

    return w, scale


def make_failure(model, model_name, prob_sa0=0, prob_sa1=0, include_layers=None, remove_module=False):

    if not remove_module:
        # for CIFAR
        model.load_state_dict(torch.load(model_name),strict=False)
    else:
        # -------------- for imagenet --------------
        state_dict = torch.load(model_name)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        # ------------------------------------------


    for (name, weight) in model.named_parameters():
        if name not in include_layers:  # ignore layers that do not have rho
            continue

        if prob_sa0 != 0:
            weight.data = failure_to_value(weight, prob_fail=prob_sa0, value=0.0)

        if prob_sa1 != 0:
            weight.data = failure_to_value(weight, prob_fail=prob_sa1, value=1.0)


def make_two_normal_failure(model, model_name, prob_sa0=0, prob_sa1=0, include_layers=None, remove_module=False):

    if not remove_module:
        # for CIFAR
        model.load_state_dict(torch.load(model_name),strict=False)
    else:
        # -------------- for imagenet --------------
        state_dict = torch.load(model_name)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        # ------------------------------------------

    for (name, weight) in model.named_parameters():
        if name not in include_layers:  # ignore layers that do not have rho
            continue

        maximum_value = torch.max(weight).cpu().detach().numpy().item()
        minimum_value = torch.min(weight).cpu().detach().numpy().item()

        w1, w2 = convert_to_two_normal_crossbar(weight)

        if prob_sa0 != 0:
            w1 = failure_to_value(w1, prob_fail=prob_sa0, value=0)
            w2 = failure_to_value(w2, prob_fail=prob_sa0, value=0)

        if prob_sa1 != 0:
            w1 = failure_to_value(w1, prob_fail=prob_sa1, value=maximum_value)
            w2 = failure_to_value(w2, prob_fail=prob_sa1, value=minimum_value)

        weight.data = convert_normal_to_single_crossbar(w1, w2)

device = torch.device("cuda")
def make_two_differential_crossbar_failure(model, model_name, prob_sa0=0, prob_sa1=0, include_layers=None, remove_module=False):
    if not remove_module:
        #state_dict = torch.load(model_name)
        #print(state_dict)
        #model = AutoModelWithLMHead.from_pretrained(model_name)
        #state_dict = torch.load(model_name)
        #state_dict = state_dict.get('model', state_dict)
        #model.load_state_dict(state_dict)
        #model.to(device)
        model.load_state_dict(torch.load(model_name),strict=False)
    else:
        # -------------- for imagenet --------------
        state_dict = torch.load(model_name)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        # ------------------------------------------


    for (name, weight) in model.named_parameters():
        if name not in include_layers:  # ignore layers that do not have rho
            continue

        scale, w1, w2 = convert_to_two_differential_crossbar(weight)

        if prob_sa0 != 0:
            w1 = failure_to_value(w1, prob_fail=prob_sa0, value=0)
            w2 = failure_to_value(w2, prob_fail=prob_sa0, value=0)

        if prob_sa1 != 0:
            w1 = failure_to_value(w1, prob_fail=prob_sa1, value=1)
            w2 = failure_to_value(w2, prob_fail=prob_sa1, value=1)

        weight.data = convert_differential_to_single_crossbar(scale, w1, w2)





def make_offset_crossbar_failure(model, model_name, prob_sa0=0, prob_sa1=0, include_layers=None, remove_module=False):

    if not remove_module:
        # for CIFAR
        model.load_state_dict(torch.load(model_name))
    else:
        # -------------- for imagenet --------------
        state_dict = torch.load(model_name)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        # ------------------------------------------


    for (name, weight) in model.named_parameters():
        if name not in include_layers:  # ignore layers that do not have rho
            continue

        maximum_value = torch.max(torch.abs(weight)).cpu().detach().numpy().item()
        minimum_value = -1.0 * maximum_value

        if prob_sa0 != 0:
            weight.data = failure_to_value(weight, prob_fail=prob_sa0, value=minimum_value)

        if prob_sa1 != 0:
            weight.data = failure_to_value(weight, prob_fail=prob_sa1, value=maximum_value)


