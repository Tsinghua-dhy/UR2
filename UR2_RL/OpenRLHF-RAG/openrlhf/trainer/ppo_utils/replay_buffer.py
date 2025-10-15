import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

import time
from .experience_maker import Experience
import ray
import numpy as np

@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (A)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "retrieve_mask",
    )
    # print("key-check"*20)
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            # print("None: ",key)
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        # print(key)
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
            # print("A tensor: ",key)
        else:
            pass
            # print("Not tensor: ",key) #都不是tensor
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    # print("key-check"*20)
    # kill
    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        if isinstance(v, torch.Tensor):
            vals = torch.unbind(v)
            assert batch_size == len(vals)
#        else:
            # 否则直接假设是 list 或 tuple
#            vals = v
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    # print("seq-type:",type(items[0].sequences))
    # print("log-type:",type(items[0].action_log_probs))
    # print("adv-type:",type(items[0].advantages))
    # print("mask-type:",type(items[0].retrieve_mask))
    # print("seq-len:",len(items[0].sequences))
    # print("log-len:",len(items[0].action_log_probs))
    # print("adv-log:",len(items[0].advantages))
    # print("mask-len:",len(items[0].retrieve_mask))
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "retrieve_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
#        if key != "format_reward" and key != "type":
            vals = torch.tensor([item.info[key] for item in items])
            kwargs["info"][key] = vals
#        else:
#            kwargs["info"][key] = [item.info[key] for item in items]
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad],
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            # print('item-check-begin'*5)
            # print("item-seq-len",len(item.sequences))
            # print("item-seq",item.sequences)
            # print("item-adv-len",len(item.advantages))
            # print("item-adv",item.advantages)
            # print("item-retrieve-mask",item.retrieve_mask)
            # print('item-check-final'*5)
            # kill
            # item_adv = getattr(item, "advantages")
            # item_action_mask = (item_adv != 0).float()
            # items.append(item_adv)
            # action_masks.append(item_action_mask)

            items.append(getattr(item, "advantages"))
            action_masks.append(item.retrieve_mask)

        # print("len_itmes:",len(items)) #更新一次用的prompt？#TODO 看一下这个的含义
        # list_data_adv = items[0].tolist()
        # print("item-adv[0]: ",list_data_adv, flush=True)
        # list_data_mask = action_masks[0].tolist()
        # print("item-mask[0]: ",list_data_mask, flush=True)
        # print("item-mask_ori[0]: ",action_masks_ori[0], flush=True)
        # kill

        # time.sleep(10)
        # list_data = items[0].tolist()
        # time.sleep(10)
        # print("buffer-itmes[0]:",list_data)


        items_vector = torch.cat(items).float().flatten()
        # time.sleep(10)
        # print("257-sht-bude"*10)
        # time.sleep(10)
        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            # time.sleep(10)
            # print("buffer-mask-try[0]:",action_masks[0])
            # time.sleep(10)
            # print("buffer-adv-try[0]:")
            # time.sleep(10)
            # print("items[0]-len:",len(items))
            # print(items[0].cpu().numpy())
            # time.sleep(10)
            # print(items[0])
            # time.sleep(10)
            # print("265-sht-bude"*10)
            # time.sleep(10)
            # print("266-sht-bude"*10)
            # # time.sleep(10)
            # print("267-sht-bude"*10)
            # # time.sleep(10)
            # print("buffer-mask-try[1]:",action_masks[1])

            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()


        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        # print("items_len",len(items))
        # print("items_vector_len",len(items_vector))
        # print("all_sum:",all_sum)
        # print("num_actions:",num_actions)
        # print("all_count:",all_count)
        # list_data = items[0].tolist()
        # print("buffer-items[0]:",list_data )

        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()
        # print("buffer-rstd:",rstd)
        # print("buffer-mean:",mean)

        # print("type items[0].seq:",type(self[0].sequences))
        # print("type items[0].log:",type(self[0].action_log_probs))
        # print("type items[0].adv:",type(self[0].advantages))
        # print("type items[0].mask:",type(self[0].retrieve_mask))
        # print("type items[0].seq:",len(self[0].sequences))
        # print("type items[0].log:",len(self[0].action_log_probs))
        # print("type items[0].adv:",len(self[0].advantages))
        # print("type items[0].mask:",len(self[0].retrieve_mask))
        # time.sleep(10)
        # list_data_1 = ((items[0] - mean) * rstd)
        # time.sleep(10)
        # list_data = list_data_1.tolist()
        # time.sleep(10)
        # print("buffer-itmes[0]:",list_data)
        # time.sleep(10)
        # kill
        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
            # if i==0:
            #     time.sleep(5)
            #     list_data = ((items[i] - mean) * rstd).tolist()
            #     time.sleep(5)
            #     print("buffer-itmes[0]:",list_data)


