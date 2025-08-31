import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random
random.seed(42)
#sht update
import ray
import torch
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
import copy
import requests
import time
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
import re
from collections import defaultdict
logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


# Update 1229 For GRPO
def conditional_cat(attr1, attr2):
    if attr1 is not None and attr2 is not None:
        if isinstance(attr1, torch.Tensor):
            op = lambda x, y: torch.cat((x, y), dim=0)
        else:
            op = lambda x, y: x + y
        return op(attr1, attr2)
    return None


# End of Update


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    retrieve_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None


    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.retrieve_mask = to(self.retrieve_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.retrieve_mask = pin_memory(self.retrieve_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self

    # CZP Update 1229 For GRPO
    def __add__(self, other):
        if not isinstance(other, Experience):
            return NotImplemented

        info = {}
        for k in self.info.keys():
            info[k] = conditional_cat(self.info[k], other.info[k])

        return Experience(
            sequences=conditional_cat(self.sequences, other.sequences),
            action_log_probs=conditional_cat(self.action_log_probs, other.action_log_probs),
            values=conditional_cat(self.values, other.values),
            returns=conditional_cat(self.returns, other.returns),
            advantages=conditional_cat(self.advantages, other.advantages),
            attention_mask=conditional_cat(self.attention_mask, other.attention_mask),
            action_mask=conditional_cat(self.action_mask, other.action_mask),
            retrieve_mask=conditional_cat(self.retrieve_mask, other.retrieve_mask),
            # retrieve_mask=None,
            info=info,
            kl=conditional_cat(self.kl, other.kl),
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    # End of Update


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    retrieve_num: torch.Tensor
    pure_response_length: torch.Tensor


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        # CZP Update 1229 For GRPO
        self.args = strategy.args
        print(self.args)
        # End of Update

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # generate responses
        # print("all_prompts-len: ", len(all_prompts))
        # print("all_prompts[0:10]: ", all_prompts[0:10])
        # kill
        samples_list = self.generate_samples(all_prompts, **generate_kwargs) #vllm生成，这里会给action_mask赋值，改这里 #TODO!!
        torch.distributed.barrier()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))
        #在make_experience中，experience处理计算KL时还是packing的形式，处理完后unpacking成了普通形式
        # print("make_list_experiences size: ", len(experiences)) #64，每一个是一个experience，里面的内容都是unpacked的
        # print("seq[0]-len", len(experiences[0].sequences))
        # # print("seq[0]", experiences[1].sequences)
        # print("seq[1]-len", len(experiences[1].sequences))
        # print("mask[0]-len", len(experiences[0].retrieve_mask))
        # # print("mask[0]", experiences[1].retrieve_mask)
        # print("mask[1]-len", len(experiences[1].retrieve_mask))

        # print("line 241- experiences size: ", len(experiences[0]))
        # kill

        experiences, rewards = self.process_experiences(experiences) #这一步在干什么 —— 把experience拼成一个整体，方便后续计算mean
        # print("line 246- experiences size: ", len(experiences)) #1，那下面的for循环是为了什么 —— 应该是没有改


        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")

            # Testing
            num_actions = experience.info["num_actions"]

            # Update 1229 For GRPO
            if self.advantage_estimator == "group_norm":
                assert self.args.n_samples_per_prompt > 1, "group_norm requires n_samples_per_prompt > 1"

                # multi-reward reward size: [bz * n_samples, reward_kinds]
                print(f"reward: {reward.size()}")
                reward = reward.reshape(-1, self.args.n_samples_per_prompt, 2)
                label_reward = reward[:, :, 0].clone()

                total_counts = label_reward.size(1)
                print(f"reward: {reward.size()}, label_reward: {label_reward.size()}")

                acc_info = (label_reward == 1).float().reshape(-1)
                experience.info["acc"] = acc_info

                length_reward = reward[:, :, 1].clone()

                reward_bk = reward.clone()
                reward = length_reward + label_reward

                # Original Version
                reward = reward.reshape(-1, self.args.n_samples_per_prompt)
                # Update

                mean = reward.mean(1, keepdim=True)
                std = reward.std(1, keepdim=True)

                experience.info["reward"] = reward.detach().clone().reshape(-1)
                reward = (reward - mean) / (std + 1e-8)
                # reward = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-8)
                # End update
                reward = reward.reshape(-1)
                print(f'after normalize reward: {reward.size()}, length_reward: {length_reward.size()}, label_reward: {label_reward.size()}')

                # experience.info["reward"] = reward.detach().clone()

                experience.info["length_reward"] = length_reward.reshape(-1)
                experience.info["label_reward"] = label_reward.reshape(-1)

                complete_mask = label_reward >= -1
                print(f'label_reward size: {label_reward.size()}')
                print(f'complete_mask size: {complete_mask.size()}')
                count_of_completion = complete_mask.sum(dim=1)

                # 计算概率并转换为张量
                completion_ratio = (count_of_completion / total_counts).to(torch.cuda.current_device())
                completion_ratio = completion_ratio.unsqueeze(1)
                completion_ratio = completion_ratio.expand_as(label_reward).reshape(-1)
                experience.info["completion_ratio"] = completion_ratio

                complete_length = experience.info["response_length"].clone().to(torch.cuda.current_device())
                flatten_mask = complete_mask.clone().reshape(-1).to(torch.cuda.current_device())
                complete_length[~flatten_mask] = 0
                complete_length = complete_length / completion_ratio
                experience.info["completion_length"] = complete_length

            # End of Update
            # 只需要在这里把KL 中retrieve_doc对应的位置给mask掉，后面计算reward，returns，advantages，loss自然都会是0
            # 要检验一下是否如此
            # 先使用special token作为标志符进行mask，然后替换成```

            experience_num = len(experience.sequences)

            # for h in range(experience_num):
            #     kl = torch.clone(experience.kl[h])
            #     seq = experience.sequences[h][-len(kl):]
            #     start_indices = (seq == 151657).nonzero(as_tuple=True)[0]
            #     end_indices = (seq == 151658).nonzero(as_tuple=True)[0]
            #     assert len(start_indices) == len(end_indices), "KL: start_indices and end_indices should have the same length"
            #     for start, end in zip(start_indices, end_indices):
            #         kl[start:end + 1] = 0
            #     experience.kl[h] = kl

            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                retrieve_mask=experience.retrieve_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            # list_data_seq = reward[0].tolist()
            # print("maker-reward[0]-try: ", list_data_seq)

            # kill

            for k in experience.info:
                if isinstance(experience.info[k], torch.Tensor):
                    print("end of make list: ", k, experience.info[k].size(), experience.info[k])
                    # print(experience.info[k])
                elif isinstance(experience.info[k], list):
                    print("end of make list: ", f"{k} list len: {len(experience.info[k])}")

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    experience.retrieve_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)

                experience_num = len(experience.sequences)

                # for h in range(experience_num):
                #     advantages = torch.clone(experience.advantages[h])
                #     seq = experience.sequences[h][-len(advantages):]
                #     start_indices = (seq == 151657).nonzero(as_tuple=True)[0]
                #     end_indices = (seq == 151658).nonzero(as_tuple=True)[0]
                #     assert len(start_indices) == len(end_indices), "advantages: start_indices and end_indices should have the same length"
                #     for start, end in zip(start_indices, end_indices):
                #         advantages[start:end + 1] = 0
                #     experience.advantages[h] = advantages

                # list_data_seq = experience.sequences[0].tolist()
                # print("maker-experience.seq-try: ", list_data_seq)

                # list_data_mask = experience.retrieve_mask[0].tolist()
                # print("maker-experience.retrieve_mask-try: ", list_data_mask)

                # list_data_kl = experience.kl[0].tolist()
                # print("maker-experience.kl-try: ", list_data_kl)

                # list_data_adv = experience.advantages[0].tolist()
                # print("maker-experience.advantages-try-0: ", list_data_adv, flush=True)
                # # kill

                # print("sht-debug-"*10)
                # self.show_experience(experience)
                # 在这里检验一下对应位置是否是 0 即可；如果是0，那么最后loss也是不需要mask了
                # print("advantage_size: ", experience.advantages.size())
                # print("num_actions: ", num_actions)
                # print("sequences_size: ", experience.sequences.size())
                # print("sht-debug-"*10)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        # print("experiences size: ", len(experiences))
        # print("experiences type: ", type(experiences))
        # print("experiences[0].seq size: ", len(experiences[0].sequences))

        return experiences


    @torch.no_grad()
    def show_experience(self, experience: Experience):
        if isinstance(experience.action_log_probs, list):
            print(
                "list len",
                len(experience.action_log_probs),
                "action_log_probs: ",
                experience.action_log_probs[0].size(),
            )
            # print("action_log_probs: ", experience.action_log_probs[0].size())
        else:
            print("tensor action_log_probs: ", experience.action_log_probs.size())
        if isinstance(experience.sequences, list):
            torch.set_printoptions(threshold=500)
            list_data = experience.sequences[0].tolist()
            print("experience.sequences-try: ", list_data)
            # for k in range(0,90,9):
            #     print(f"experience.sequences_content:1:{k}:", experience.sequences[0][k:k+9])
            # for g in range(0,90,9):
            #     print(f"experience.sequences_content:2:{g}:", experience.sequences[1][g:g+9])

            # for l in range(0,90,9):
            #     print(f"experience.sequences_content:4:{l}:", experience.sequences[3][l:l+9])

            # print("experience.sequences_content: ", experience.sequences[1][40:89])
            # print("experience.sequences_content: ", experience.sequences[2][40:89])
            print("experience.sequences-1", experience.sequences[0].size(), "list len", len(experience.sequences))
            # print("experience.sequences-2", experience.sequences[1].size(), "list len", len(experience.sequences))
            # print("experience.sequences-3", experience.sequences[2].size(), "list len", len(experience.sequences))
            # print("experience.sequences-4", experience.sequences[3].size(), "list len", len(experience.sequences))
            # print("experience.sequences-5", experience.sequences[4].size(), "list len", len(experience.sequences))
            # print("experience.sequences-6", experience.sequences[5].size(), "list len", len(experience.sequences))
            # print("experience.sequences-7", experience.sequences[6].size(), "list len", len(experience.sequences))

            torch.set_printoptions(threshold=10)
        elif experience.sequences is not None:
            print("tensor experience.sequences", experience.sequences.size())
        else:
            print("experience.sequences is None")
        if isinstance(experience.values, list):
            print("experience.values", experience.values[0].size(), "list len", len(experience.values))
            # print("experience.values", experience.values[0].size())
        elif experience.values is not None:
            print("tensor experience.values", experience.values.size())
        else:
            print(f"experience.values is None")
        if isinstance(experience.returns, list):
            print("experience.returns: ", experience.returns[0].size(), "list len", len(experience.returns))
            # print("experience.returns: ", experience.returns[0].size())
        elif experience.returns is not None:
            print("tensor experience.returns: ", experience.returns.size())
        else:
            print(f"experience.returns is None")
        if isinstance(experience.advantages, list):
            list_data = experience.advantages[0].tolist()
            print("experience.advantages-try: ", list_data)
            # for k in range(0,90,9):
            #     print(f"experience.advantages_content:1:{k}:", experience.advantages[0][k:k+9])
            # for g in range(0,90,9):
            #     print(f"experience.advantages_content:2:{g}:", experience.advantages[1][g:g+9])
            # for l in range(0,90,9):
            #     print(f"experience.advantages_content:4:{l}:", experience.advantages[3][l:l+9])

            print("experience.advantages-1: ", experience.advantages[0].size(), "list len", len(experience.advantages))
            # print("experience.advantages-2: ", experience.advantages[1].size(), "list len", len(experience.advantages))
            # print("experience.advantages-3: ", experience.advantages[2].size(), "list len", len(experience.advantages))
            # print("experience.advantages-4: ", experience.advantages[3].size(), "list len", len(experience.advantages))
            # print("experience.advantages-5: ", experience.advantages[4].size(), "list len", len(experience.advantages))
            # print("experience.advantages-6: ", experience.advantages[5].size(), "list len", len(experience.advantages))
            # print("experience.advantages-7: ", experience.advantages[6].size(), "list len", len(experience.advantages))

            # print("experience.advantages: ", experience.advantages[0].size())
        elif experience.advantages is not None:
            print("tensor experience.advantages: ", experience.advantages.size())
        else:
            print(f"experience.advantages is None")
        if isinstance(experience.attention_mask, list):
            print(
                "experience.attention_mask: ",
                experience.attention_mask[0].size(),
                "list len",
                len(experience.attention_mask),
            )
            # print("experience.attention_mask: ", experience.attention_mask[0].size())
        elif experience.attention_mask is not None:
            print("tensor experience.attention_mask: ", experience.attention_mask.size())
        else:
            print(f"experience.attention_mask is None")
        if isinstance(experience.action_mask, list):
            print(
                "experience.action_mask: ", experience.action_mask[0].size(), "list len", len(experience.action_mask)
            )
            # print("experience.action_mask: ", experience.action_mask[0].size())
        elif experience.action_mask is not None:
            print("tensor experience.action_mask: ", experience.action_mask.size())
        else:
            print(f"experience.action_mask is None")
        if isinstance(experience.kl, list):
            print("experience.kl: ", experience.kl[0].size(), "list len", len(experience.kl))
            # print("experience.kl: ", experience.kl[0].size())
        elif experience.kl is not None:
            print("tensor experience.kl: ", experience.kl.size())
        else:
            print(f"experience.kl is None")

        for k in experience.info:
            if isinstance(experience.info[k], torch.Tensor):
                print(k, experience.info[k].size(), experience.info[k])
                # print(experience.info[k])
            elif isinstance(experience.info[k], list):
                print(f"{k} list len: {len(experience.info[k])}, first: {experience.info[k][0]}")


    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        kill
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        kill
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # print("sht-debug"*20)
        # print(len(sequences))
        # print(len(sequences[0]))
        # print("sht-debug"*20)
        # kill
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # print("sht-debug-"*10)
            # print(len(queries))
            # print(queries[0:1])
            # print(self.remote_rm_url)
            # print("sht-debug-"*10)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, retrieve_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            for experience in experiences:
                experience.info['acc_info'] = (experience.info["reward"] == 1).float().reshape(-1)
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            # acc_info = (rewards == 1).float()
            # expe
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards

        # CZP Update 1229 For GRPO
        if self.advantage_estimator in ["group_norm"]:
            return [sum(experiences)], [experience.info["reward"] for experience in [sum(experiences)]]
        # End of Update

        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        kill
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        retrieve_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            # print(len(rewards))
            # print(type(retrieve_mask[0]))
            # kill
            returns = []
            for p, r in enumerate(rewards):
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, retrieve_mask[p].unsqueeze(0), gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # print("cumulative-reward:",rewards)
        # kill
        # Calculate returns by accumulating
        # discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        # print("cumulative-returns-size: ", returns.size())
        # print("cumulative-retrieve_mask-size: ", retrieve_mask.size())
        # print("cumulative-retrieve_mask-size[0]: ", len(retrieve_mask[0]))
        if retrieve_mask is not None:
            returns = returns * retrieve_mask
        # print("cumulative-returns-size: ", returns.size())
        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }


        experiences = super().make_experience_list(all_prompts, **generate_kwargs) #一个list，只有[0]是一个tensor，装64个experience
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        # return self._generate_vllm(all_prompts, **generate_kwargs)
        return self._generate_vllm_with_retrieve_new_new(all_prompts, **generate_kwargs)


    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        # kill
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        retrieve_mask = samples.retrieve_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        # print("sht-debug"*20)
        # print('sequences')
        # print(len(sequences)) #1
        # print(len(sequences[0])) #868
        # print("sht-debug"*20)


        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            for rm in self.remote_rm_url:
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                # kills
                r_refs.append(r)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        # print("sht-debug"*20)
        # print("action_log_probs")
        # print(len(action_log_probs))
        # print(len(action_log_probs[0]))
        # print("sht-debug"*20)
        # kill
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            retrieve_mask=retrieve_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask,retrieve_mask=retrieve_mask, dim=-1,)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            retrieve_mask = unpacking_samples(retrieve_mask, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.zeros(len(kl), device=device)  # 预先创建一个张量，长度为 kl 的长度
            for i, each_kl in enumerate(kl):
                # kl_mean[i] = each_kl.mean()
                kl_mean[i] = masked_mean(each_kl, action_mask,retrieve_mask=retrieve_mask[i], dim=-1,)


            # kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)
        # print("samples-seq-len: ", len(sequences))
        # print("samples-seq[0]-len: ", len(sequences[0]))
        # print("samples-seq[1]-len: ", len(sequences[1]))
        # print("samples-seq[0]",sequences[0])

        # print("samples-retrieve-mask-len: ", len(retrieve_mask))
        # print("samples-retrieve-mask[0]-len: ", len(retrieve_mask[0]))
        # print("samples-retrieve-mask[1]-len: ", len(retrieve_mask[1]))
        # print("samples-retrieve-mask[0]",retrieve_mask[0])
        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "retrieve_num": samples.retrieve_num,
            "pure_response_length": samples.pure_response_length,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            retrieve_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm_with_retrieve(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        url_wiki = "http://0.0.0.0:5003/queries"
        from vllm import SamplingParams
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        # print("all_prompts-len:",all_prompts)
        # print("all_prompts[0:4]:",all_prompts[0:4])
        # kill

        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        prompts_w_idx_dict = []
        for idx_w_prompt in all_prompts:
            idx, prompt = idx_w_prompt.split("<|idx_prompt_split|>")
            prompts_w_idx_dict.append({"idx": idx, "prompt": prompt})
        # 从这里开始加检索功能
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</query>"]
        batch_size = (len(prompts_w_idx_dict) + len(llms) - 1) // len(llms)
        # print("llms-len:",len(llms))
        all_outputs = []
        for i, llm in enumerate(llms):
            # prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]

            idx_w_prompt_part = prompts_w_idx_dict[i * batch_size : (i + 1) * batch_size]
            # print("idx_w_prompt_part:",idx_w_prompt_part)
            data_keys = ["prompt" , "idx"]
            ds = Dataset.from_dict({key: [d[key] for d in idx_w_prompt_part] for key in data_keys})
            # print("ds:",ds)

            finished_all_list=[]
            continued_answer = copy.deepcopy(idx_w_prompt_part)

            for t in range(11):
                finished_texts = []
                continued_texts = []
                sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=512, stop=stop_tokens)

                outputs_ray = llm.generate.remote(ds['prompt'], sampling_params)
                outputs = ray.get(outputs_ray)
                query_list=[]
                for q, output in enumerate(outputs):

                    prompt = output.prompt
                    idx = continued_answer[q]["idx"]
                    stop_reason = output.outputs[0].stop_reason
                    generated_text = output.outputs[0].text

                    if "prompt_ids" not in continued_answer[q]:
                        input_token_ids = list(output.prompt_token_ids)
                    else:
                        input_token_ids = continued_answer[q]["prompt_ids"]
                    # if "response_ids" not in continued_answer[q]:
                    #     output_token_ids = list(output.outputs[0].token_ids)
                    # else:
                    #     output_token_ids= continued_answer[q]["response_ids"].extend(list(output.outputs[0].token_ids))

                    all_token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                    output_token_ids = all_token_ids[len(input_token_ids):]
                    # print(f"output_token_ids-{q}: {output_token_ids}")
                    if t == 9: #检索次数太多了，直接停掉，就是未完成
                        original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                                }
                        finished_texts.append(original_data)
                        continue

                    if "<query>" in generated_text and stop_reason=="</query>":
                        query = generated_text.split("<query>")[-1].split("</query>")[0]
                        query = query.replace('"',"")
                        if query: #开始检索
                            query_list.append(query)
                            original_data = {
                                "idx":idx,
                                "prompt":prompt + generated_text.strip(),
                                "prompt_ids":input_token_ids,
                                "response_ids":output_token_ids,
                                }
                            continued_texts.append(original_data)
                            # print("(1)ori-dict:",original_data)
                        else: #这个是query没有按照规定格式的 ，直接停止了，之后可能需要优化
                            original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                                }
                            finished_texts.append(original_data)
                            # print("(2)ori-dict:",original_data)
                    else: #生成结束
                        original_data = {
                        "idx":idx,
                        "prompt_ids":input_token_ids,
                        "response_ids":output_token_ids,
                        }
                        finished_texts.append(original_data)
                        # print("(3)ori-dict:",original_data)

                assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
                # print("query-list:",query_list)
                if len(query_list)!=0:
                    topk = 3
                    response = requests.post(url_wiki, json={"queries": query_list, "k": topk})
                    if response.status_code == 200:
                        result = response.json()
                        answers = result["answers"]
                        for k in range(len(answers)):
                            retrieve_docs = answers[k]
                            continued_text_now = copy.deepcopy(continued_texts[k])
                            if len(retrieve_docs)>0:
                                doc_content_list = []
                                for j in range(len(retrieve_docs)):
                                    doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                    doc_content_list.append(f"({j+1}){doc_now}\n")
                                doc_content = ''.join(doc_content_list)
                            else:
                                doc_content = "None"

                            continued_text_now["prompt"] = continued_text_now["prompt"] + "</query>"+ "\n\nThe retrieved content are:\n<tool_call>\n"  +  doc_content + "\n</tool_call>\n\n"
                            continued_texts[k] = continued_text_now
                            # print("continued_text_now:",continued_text_now)
                    else:
                        raise Exception("Error in response: the status code is not 200!")

                finished_all_list.extend(finished_texts)
                if len(continued_texts)==0:
                    if len(finished_all_list) != len(idx_w_prompt_part):
                        time.sleep(20)
                        print("finished_all_list:",finished_all_list)
                        print("idx_w_prompt_part:",idx_w_prompt_part)
                        time.sleep(20)
                    assert len(finished_all_list) == len(idx_w_prompt_part) , "Error in len of finished_all_list and idx_w_prompt_part"
                    all_outputs.append(finished_all_list)
                    break
                else:
                    data_keys_again = continued_texts[0].keys()
                    ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                    continued_answer = copy.deepcopy(continued_texts)

        all_outputs_cat = [item for sublist in all_outputs for item in sublist]
        if len(all_outputs_cat) != len(prompts_w_idx_dict):
            time.sleep(20)
            print("all_outputs_cat:",all_outputs_cat)
            print("prompts_w_idx_dict:",prompts_w_idx_dict)
            time.sleep(20)
        assert len(all_outputs_cat) == len(prompts_w_idx_dict), "Error in len of all_outputs and prompts_w_idx_dict"
        all_outputs_sorted = sorted(all_outputs_cat, key=lambda x: x['idx'])
        # for l in range(len(all_outputs_sorted)):
        #     print(f"all_outputs_sorted-{l}: {all_outputs_sorted[l]}")


        # new_tokens = [151657,   1986,    374,    264,   1273,     13,    358,   3900,    419,
        #     949,    374,   1083,    220,     15,    518,    279,  12159,   2309,
        #     315,  22146, 151658]
        # token_ids = list(all_outputs[0].outputs[0].token_ids)
        # insert_position = len(token_ids) - 4
        # token_ids = token_ids[:insert_position + 1] + new_tokens + token_ids[insert_position + 1:]
        # insert_position_100 = len(token_ids) - 100
        # token_ids = token_ids[:insert_position_100 + 1] + new_tokens + token_ids[insert_position_100 + 1:]
        # all_outputs[0].outputs[0].token_ids = token_ids
        # print ("prompt_token_ids:",all_outputs[0].prompt_token_ids)
        # print("token_ids",all_outputs[0].outputs[0].token_ids)
        # print("output:",all_outputs[0])
        # kill
        # print("all_optputs_sorted",all_outputs_sorted)
        samples_list = []
        for i in range(0, len(all_outputs_sorted), args.micro_rollout_batch_size):
            outputs = all_outputs_sorted[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                pass
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                retrieve_mask=[]
                num_actions = []

                retrieve_num = []
                pure_response_length_lis = []
                for i, output in enumerate(outputs):
                    input_len = len(output["prompt_ids"])
                    output_len = len(output["response_ids"])
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output["prompt_ids"] + output["response_ids"])
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    response_seq = output["response_ids"]
                    retrieve_mask_now = [1]*len(response_seq)
                    start_indices = [g for g, x in enumerate(response_seq) if x == 151657]

                    end_indices = [g for g, x in enumerate(response_seq) if x == 151658]
                    if len(start_indices) == len(end_indices):
                        for start, end in zip(start_indices, end_indices):
                            for h in range(start, end + 1):  # 包括 end
                                retrieve_mask_now[h] = 0
                    else:
                        import time
                        time.sleep(5)
                        print("Warning: start_indices and end_indices should have the same length")
                        print("start_indices:",start_indices)
                        print("end_indices:",end_indices)
                        print("output-important:",output)
                        time.sleep(5)
                        for h in range(len(response_seq)):
                            retrieve_mask_now[h] = 1
                    # assert len(start_indices) == len(end_indices), "KL: start_indices and end_indices should have the same length"

                    retrieve_mask.extend(retrieve_mask_now)

                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)

                # list_data_seq = sequences.tolist()
                # print("samples-list-sequences: ", list_data_seq)
                # list_data_mask = retrieve_mask.tolist()
                # print("samples-list-retrieve_mask: ", list_data_mask)


                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask = retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        # print("samples_list:",samples_list)
        # kill
        return samples_list

    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            # ZBC Update
            if args.random_temperature:
                values = [i / 10 for i in range(5, 11)]
                sampling_params.temperature = random.choice(values)
            print("temperature", sampling_params.temperature)
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        new_tokens = [151657,   1986,    374,    264,   1273,     13,    358,   3900,    419,
            949,    374,   1083,    220,     15,    518,    279,  12159,   2309,
            315,  22146, 151658]
        token_ids = list(all_outputs[0].outputs[0].token_ids)
        insert_position = len(token_ids) - 4
        token_ids = token_ids[:insert_position + 1] + new_tokens + token_ids[insert_position + 1:]
        insert_position_100 = len(token_ids) - 100
        token_ids = token_ids[:insert_position_100 + 1] + new_tokens + token_ids[insert_position_100 + 1:]
        all_outputs[0].outputs[0].token_ids = token_ids

        # print("vllm-all_outputs-0-try:",all_outputs[0].outputs[0].token_ids) #128

        # token_ids = list(all_outputs[1].outputs[0].token_ids)
        # insert_position_100 = len(token_ids) - 100
        # token_ids = token_ids[:insert_position_100 + 1] + new_tokens + token_ids[insert_position_100 + 1:]
        # all_outputs[1].outputs[0].token_ids = token_ids

        # print("all_outputs_len:",len(all_outputs)) #128
        # print("=="*20)
        # print("all_outputs[0].prompt_token_ids",type(all_outputs[0].prompt_token_ids))
        # print("=="*20)
        # print("all_outputs[0].token_ids",type(all_outputs[0].outputs[0].token_ids))
        # print("=="*20)
        # print("all_outputs[0]",all_outputs[0])
        # kill

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                retrieve_mask = retrieve_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        retrieve_mask=retrieve_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                retrieve_mask=[]
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    response_seq = list(output.outputs[0].token_ids)
                    retrieve_mask_now = [1]*len(response_seq)
                    # start_indices = (response_seq == 151657).nonzero(as_tuple=True)[0]
                    # end_indices = (response_seq == 151658).nonzero(as_tuple=True)[0]
                    start_indices = [g for g, x in enumerate(response_seq) if x == 151657]
                    end_indices = [g for g, x in enumerate(response_seq) if x == 151658]
                    assert len(start_indices) == len(end_indices), "KL: start_indices and end_indices should have the same length"
                    for start, end in zip(start_indices, end_indices):
                        for h in range(start, end + 1):  # 包括 end
                            retrieve_mask_now[h] = 0
                    retrieve_mask.extend(retrieve_mask_now)

                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                # print("sequences_size:",sequences.size())
                # print("attention_mask_size:",attention_mask.size())
                # print("vllm-retrieve_mask_size:",retrieve_mask.size())
                # print("response_length_size:",response_length.size())
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask = retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list
    
    
    
    def _generate_vllm_with_retrieve_new(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        url_wiki = "http://0.0.0.0:5003/queries"
        from vllm import SamplingParams
        import time
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        prompts_w_idx_dict = []
        for idx_w_prompt in all_prompts:
            idx, prompt = idx_w_prompt.split("<|idx_prompt_split|>")
            prompts_w_idx_dict.append({"idx": idx, "prompt": prompt})
        # 从这里开始加检索功能
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        stop_tokens = ["</query>"]
        batch_size = (len(prompts_w_idx_dict) + len(llms) - 1) // len(llms)
        # print("llms-len:",len(llms))
        all_outputs = []
        for i, llm in enumerate(llms):
            # prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]

            idx_w_prompt_part = prompts_w_idx_dict[i * batch_size : (i + 1) * batch_size]
            # print("idx_w_prompt_part:",idx_w_prompt_part)
            data_keys = ["prompt" , "idx"]
            ds = Dataset.from_dict({key: [d[key] for d in idx_w_prompt_part] for key in data_keys})
            # print("ds:",ds)

            finished_all_list=[]
            continued_answer = copy.deepcopy(idx_w_prompt_part)

            for t in range(11):
                finished_texts = []
                continued_texts = []
                sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=512, stop=stop_tokens)

                outputs_ray = llm.generate.remote(ds['prompt'], sampling_params)
                outputs = ray.get(outputs_ray)
                query_list=[]
                for q, output in enumerate(outputs):

                    prompt = output.prompt
                    idx = continued_answer[q]["idx"]
                    stop_reason = output.outputs[0].stop_reason
                    generated_text = output.outputs[0].text

                    if "prompt_ids" not in continued_answer[q]:  #第一轮生成，最初的prompt
                        input_token_ids = list(output.prompt_token_ids)
                    else:
                        input_token_ids = continued_answer[q]["prompt_ids"]

                    if "prompt_text" not in continued_answer[q]:  #第一轮生成，最初的prompt
                        input_token_text = output.prompt
                    else:
                        input_token_text = continued_answer[q]["prompt_text"]


                    if "retrieve_mask" not in continued_answer[q]:  #第一轮生成，第一轮的gen_token的mask(全是1)
                        if stop_reason =="</query>":
                            prompt_wo_doc_ids = self.tokenize_fn(output.outputs[0].text + "</query>\n\n", 4096, padding=False)["input_ids"]
                            retrieve_mask_gen =[1]*len(prompt_wo_doc_ids)
                        else:
                            gen_tokens_len = len(output.outputs[0].token_ids)
                            retrieve_mask_gen =[1]*gen_tokens_len

                    else :#因为外部检索所以进行这个分支，需要补上检索的doc的和正常的生成的（Note：doc的是在这里补上的）
                        retrieve_mask_gen = continued_answer[q]["retrieve_mask"]
                        prompt_wo_doc_ids = self.tokenize_fn(continued_answer[q]["prompt_wo_doc"],  4096, padding=False)["input_ids"]
                        doc_tokens_len = len(list(output.prompt_token_ids)) - len(prompt_wo_doc_ids)
                        if stop_reason =="</query>":
                            gen_tokens_len =len(self.tokenize_fn(output.outputs[0].text + "</query>\n\n",  4096, padding=False)["input_ids"])
                        else:
                            gen_tokens_len = len(output.outputs[0].token_ids)
          
                        retrieve_mask_gen =retrieve_mask_gen + [0]*doc_tokens_len + [1]*gen_tokens_len

                    all_token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                    output_token_ids = all_token_ids[len(input_token_ids):]

                    all_token_text = output.prompt+output.outputs[0].text
                    output_token_text = all_token_text[len(input_token_text):]
                    # print("prompt-check",[all_token_text,output_token_ids])
                    # time.sleep(15)
                    # kill
                    # print(f"output_token_ids-{q}: {output_token_ids}")
                    if t == 1: #检索次数太多了，直接停掉，就是未完成
                        original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "prompt_text":input_token_text,
                            "response_ids":output_token_ids, #.append(151645)
                            "response_text":output_token_text,
                            "retrieve_mask": retrieve_mask_gen
                                }

                        finished_texts.append(original_data)
                        continue    

                    if "<query>" in generated_text and stop_reason=="</query>":
                        query = generated_text.split("<query>")[-1].split("</query>")[0]
                        query = query.replace('"',"").strip()
                        if query: #开始检索
                            query_list.append(query)
                            original_data = {
                                "idx":idx,
                                "prompt":prompt + generated_text.strip(),
                                "prompt_ids":input_token_ids,
                                "prompt_text":input_token_text,
                                "response_ids":output_token_ids,
                                "response_text":output_token_text,
                                "retrieve_mask": retrieve_mask_gen
                                }
                            continued_texts.append(original_data)
                            # print("(1)ori-dict:",original_data)
                        else: #这个是query没有按照规定格式的 ，直接停止了，之后可能需要优化
                            original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "prompt_text":input_token_text,
                            "response_ids":output_token_ids,
                            "response_text":output_token_text,
                            "retrieve_mask": retrieve_mask_gen
                                }
                            finished_texts.append(original_data)
                            # print("(2)ori-dict:",original_data)
                    else: #生成结束
                        original_data = {
                        "idx":idx,
                        "prompt_ids":input_token_ids,
                        "prompt_text":input_token_text,
                        "response_ids":output_token_ids,
                        "response_text":output_token_text,
                        "retrieve_mask": retrieve_mask_gen
                        }
                        finished_texts.append(original_data)

                assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
                # print("query-list:",query_list)
                if len(query_list)!=0:
                    topk = 3
                    response = requests.post(url_wiki, json={"queries": query_list, "k": topk})
                    if response.status_code == 200:
                        result = response.json()
                        answers = result["answers"]
                        for k in range(len(answers)):
                            retrieve_docs = answers[k]
                            continued_text_now = copy.deepcopy(continued_texts[k])
                            if len(retrieve_docs)>0:
                                doc_content_list = []
                                for j in range(len(retrieve_docs)):
                                    doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                    doc_content_list.append(f"({j+1}){doc_now}\n")
                                doc_content = ''.join(doc_content_list)
                            else:
                                doc_content = "None"

                            continued_text_now["prompt_wo_doc"] = continued_text_now["prompt"] + "</query>\n\n" 
                            continued_text_now["prompt"] = continued_text_now["prompt"] + "</query>\n\n"+ "<|begin_of_documents|>\n" +  doc_content + "<|end_of_documents|>\n\n"
                        
                            continued_texts[k] = continued_text_now
                            # print("continued_text_now:",continued_text_now)
                    else:
                        raise Exception("Error in response: the status code is not 200!")

                finished_all_list.extend(finished_texts)
                if len(continued_texts)==0:
                    assert len(finished_all_list) == len(idx_w_prompt_part) , "Error in len of finished_all_list and idx_w_prompt_part"
                    all_outputs.append(finished_all_list)
                    break
                else:
                    data_keys_again = continued_texts[0].keys()
                    ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                    continued_answer = copy.deepcopy(continued_texts)

        all_outputs_cat = [item for sublist in all_outputs for item in sublist]

        assert len(all_outputs_cat) == len(prompts_w_idx_dict), "Error in len of all_outputs and prompts_w_idx_dict"
        all_outputs_sorted = sorted(all_outputs_cat, key=lambda x: x['idx'])
        for l in range(len(all_outputs_sorted)):
            if len(all_outputs_sorted[l]["response_ids"]) !=len(all_outputs_sorted[l]["retrieve_mask"]):
                time.sleep(3)
                print("Important-bug")
                time.sleep(6)
                print(f"response_ids-bug-{l}",[all_outputs_sorted[l]["response_ids"],all_outputs_sorted[l]["retrieve_mask"],all_outputs_sorted[l]["response_text"]])
                time.sleep(15)
                # print(f"mask_ids-bug-{l}",all_outputs_sorted[l]["retrieve_mask"])
                time.sleep(3)
                print(f"res-mask_len-{l}:{len(all_outputs_sorted[l]['response_ids'])-len(all_outputs_sorted[l]['retrieve_mask'])}")
                time.sleep(3)
                kill
            # assert len(all_outputs_sorted[l]["response_ids"]) ==len(all_outputs_sorted[l]["retrieve_mask"]), "Error in len of response_ids and retrieve_mask"

        samples_list = []
        for i in range(0, len(all_outputs_sorted), args.micro_rollout_batch_size):
            outputs = all_outputs_sorted[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                pass
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                retrieve_mask=[]
                num_actions = []

                retrieve_num = []
                pure_response_length = []
                for i, output in enumerate(outputs):
                    # time.sleep(2)
                    # print(f"packed-check-{i}:{output['response_ids']}")
                    input_len = len(output["prompt_ids"])
                    output_len = len(output["response_ids"])
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output["prompt_ids"] + output["response_ids"])
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    response_seq = output["response_ids"]
                    retrieve_mask_now = [1]*len(response_seq)

                    start_indices = [g for g, x in enumerate(response_seq) if x == 151657]
                    end_indices = [g for g, x in enumerate(response_seq) if x == 151658]
                    if len(start_indices) == len(end_indices):
                        for start, end in zip(start_indices, end_indices):
                            for h in range(start, end + 1):  # 包括 end
                                retrieve_mask_now[h] = 0
                    else:
                        import time
                        time.sleep(5)
                        print("Warning: start_indices and end_indices should have the same length")
                        print("start_indices:",start_indices)
                        print("end_indices:",end_indices)
                        print("output-important:",output)
                        time.sleep(5)
                        for h in range(len(response_seq)):
                            retrieve_mask_now[h] = 0
                    # assert len(start_indices) == len(end_indices), "KL: start_indices and end_indices should have the same length"

                    retrieve_mask.extend(retrieve_mask_now)
                    num_actions.append(max(1, output_len))
                    retrieve_num.append(max(1, output_len))
                    pure_response_length.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                retrieve_num= torch.tensor(retrieve_num, device="cuda", dtype=torch.float)
                pure_response_length = torch.tensor(pure_response_length, device="cuda", dtype=torch.float)

                # list_data_seq = sequences.tolist()
                # print("samples-list-sequences: ", list_data_seq)
                # list_data_mask = retrieve_mask.tolist()
                # print("samples-list-retrieve_mask: ", list_data_mask)
                # print("response-length:",response_length)
                # print("total-length:",total_length)
                # kill


                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask = retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        retrieve_num = retrieve_num,
                        pure_response_length = pure_response_length

                    )
                )
        # print("samples_list:",samples_list)
        # kill
        return samples_list





    def _generate_vllm_with_retrieve_new_new(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        url_wiki = "http://0.0.0.0:5003/queries"
        from vllm import SamplingParams
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        # print("all_prompts-len:",all_prompts)
        # print("all_prompts[0:4]:",all_prompts[0:4])
        # kill

        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        prompts_w_idx_dict = []
        for idx_w_prompt in all_prompts:
            idx, prompt = idx_w_prompt.split("<|idx_prompt_split|>")
            prompts_w_idx_dict.append({"idx": idx, "prompt": prompt})
        # 从这里开始加检索功能
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        stop_tokens = ["</query>"]
        batch_size = (len(prompts_w_idx_dict) + len(llms) - 1) // len(llms)
        # print("llms-len:",len(llms))
        all_outputs = []
        for i, llm in enumerate(llms):
            # prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]

            idx_w_prompt_part = prompts_w_idx_dict[i * batch_size : (i + 1) * batch_size]
            # print("idx_w_prompt_part:",idx_w_prompt_part)
            data_keys = ["prompt" , "idx"]
            ds = Dataset.from_dict({key: [d[key] for d in idx_w_prompt_part] for key in data_keys})
            # print("ds:",ds)

            finished_all_list=[]
            continued_answer = copy.deepcopy(idx_w_prompt_part)

            for t in range(11):
                finished_texts = []
                continued_texts = []
                sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=512, stop=stop_tokens)

                outputs_ray = llm.generate.remote(ds['prompt'], sampling_params)
                outputs = ray.get(outputs_ray)
                query_list=[]
                for q, output in enumerate(outputs):

                    prompt = output.prompt
                    idx = continued_answer[q]["idx"]
                    stop_reason = output.outputs[0].stop_reason
                    generated_text = output.outputs[0].text

                    if "prompt_ids" not in continued_answer[q]:
                        input_token_ids = list(output.prompt_token_ids)
                    else:
                        input_token_ids = continued_answer[q]["prompt_ids"]

                    all_token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                    output_token_ids = all_token_ids[len(input_token_ids):]

                    if t == 6: #检索次数太多了，直接停掉，就是未完成
                        original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids.append(151645),
                                }
                        finished_texts.append(original_data)
                        continue

                    if "<query>" in generated_text and stop_reason=="</query>":
                        query = generated_text.split("<query>")[-1].split("</query>")[0]
                        query = query.replace('"',"").strip()
                        if query: #开始检索
                            query_list.append(query)
                            original_data = {
                                "idx":idx,
                                "prompt":prompt + generated_text.strip(),
                                "prompt_ids":input_token_ids,
                                "response_ids":output_token_ids,
                                }
                            continued_texts.append(original_data)
                            # print("(1)ori-dict:",original_data)
                        else: #这个是query没有按照规定格式的 ，直接停止了，之后可能需要优化
                            original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                                }
                            finished_texts.append(original_data)
                            # print("(2)ori-dict:",original_data)
                    else: #生成结束
                        original_data = {
                        "idx":idx,
                        "prompt_ids":input_token_ids,
                        "response_ids":output_token_ids,
                        }
                        finished_texts.append(original_data)
                        # print("(3)ori-dict:",original_data)

                assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
                # print("query-list:",query_list)
                if len(query_list)!=0:
                    topk = 3
                    response = requests.post(url_wiki, json={"queries": query_list, "k": topk})
                    if response.status_code == 200:
                        result = response.json()
                        answers = result["answers"]
                        for k in range(len(answers)):
                            retrieve_docs = answers[k]
                            continued_text_now = copy.deepcopy(continued_texts[k])
                            if len(retrieve_docs)>0:
                                doc_content_list = []
                                for j in range(len(retrieve_docs)):
                                    doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                    doc_content_list.append(f"({j+1}){doc_now}\n")
                                doc_content = ''.join(doc_content_list)
                            else:
                                doc_content = "None"

                            continued_text_now["prompt"] = continued_text_now["prompt"] + "</query>\n\n"+ "<|begin_of_documents|>\n" +  doc_content + "<|end_of_documents|>\n\n"
                            continued_texts[k] = continued_text_now
                            # print("continued_text_now:",continued_text_now)
                    else:
                        raise Exception("Error in response: the status code is not 200!")

                finished_all_list.extend(finished_texts)
                if len(continued_texts)==0:
                    if len(finished_all_list) != len(idx_w_prompt_part):
                        time.sleep(20)
                        print("finished_all_list:",finished_all_list)
                        print("idx_w_prompt_part:",idx_w_prompt_part)
                        time.sleep(20)
                    assert len(finished_all_list) == len(idx_w_prompt_part) , "Error in len of finished_all_list and idx_w_prompt_part"
                    all_outputs.append(finished_all_list)
                    break
                else:
                    data_keys_again = continued_texts[0].keys()
                    ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                    continued_answer = copy.deepcopy(continued_texts)

        all_outputs_cat = [item for sublist in all_outputs for item in sublist]
        if len(all_outputs_cat) != len(prompts_w_idx_dict):
            time.sleep(20)
            print("[all_outputs_cat,prompts_w_idx_dict]:",[all_outputs_cat,prompts_w_idx_dict])
            time.sleep(20)
        assert len(all_outputs_cat) == len(prompts_w_idx_dict), "Error in len of all_outputs and prompts_w_idx_dict"
        all_outputs_sorted = sorted(all_outputs_cat, key=lambda x: x['idx'])

        samples_list = []
        for i in range(0, len(all_outputs_sorted), args.micro_rollout_batch_size):
            outputs = all_outputs_sorted[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                pass
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                retrieve_mask=[]
                num_actions = []

                retrieve_num = []
                pure_response_length_lis = []
                for i, output in enumerate(outputs):
                    input_len = len(output["prompt_ids"])
                    output_len = len(output["response_ids"])
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output["prompt_ids"] + output["response_ids"])
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # response_seq = output["response_ids"]
                    # retrieve_mask_now = [1]*len(response_seq)
                    # start_indices = [g for g, x in enumerate(response_seq) if x == 151657]
                    # end_indices = [g for g, x in enumerate(response_seq) if x == 151658]
                    # for start, end in zip(start_indices, end_indices):
                    #     for h in range(start, end + 1):  # 包括 end
                    #         retrieve_mask_now[h] = 0

                    response_seq = output["response_ids"]
                    retrieve_mask_now = [1] * len(response_seq)
                    start_tokens = [27, 91, 7265, 3575, 75927, 91, 397] #<|begin_of_documents|>\n
                    end_tokens = [408, 3575, 75927, 91, 1339] #end_of_documents|>\n\n
                    is_in_masking = False
                    mask_start_idx = -1
                    start_count = 0
                    end_count = 0
                    for m in range(len(response_seq)):
                        if retrieve_mask_now[m] == 0:
                            continue
                        if not is_in_masking and m + len(start_tokens) <= len(response_seq) and response_seq[m:m + len(start_tokens)] == start_tokens:
                            start_count += 1  # 增加 start_tokens 计数
                            is_in_masking = True
                            mask_start_idx = m
                            for n in range(mask_start_idx, mask_start_idx + len(start_tokens)):
                                retrieve_mask_now[n] = 0

                        if is_in_masking:
                            retrieve_mask_now[m] = 0
                            if m + len(end_tokens) <= len(response_seq) and response_seq[m:m + len(end_tokens)] == end_tokens:
                                end_count += 1  # 增加 end_tokens 计数
                                is_in_masking = False  # 结束遮蔽
                                for u in range(m, m + len(end_tokens)):
                                    retrieve_mask_now[u] = 0
                                mask_start_idx = -1

                    print(f"response_seq_with_retrieve_mask_now,{[response_seq,retrieve_mask_now]}")
                    time.sleep(15)
                    kill
                    

                    
                    # assert len(start_indices) == len(end_indices), "KL: start_indices and end_indices should have the same length"

                    retrieve_mask.extend(retrieve_mask_now)
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)

                # list_data_seq = sequences.tolist()
                # print("samples-list-sequences: ", list_data_seq)
                # list_data_mask = retrieve_mask.tolist()
                # print("samples-list-retrieve_mask: ", list_data_mask)


                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask = retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        # print("samples_list:",samples_list)
        # kill
        return samples_list
    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
