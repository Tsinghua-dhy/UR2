from torch.utils.data import Dataset
from tqdm import tqdm
import random
def get_option_str(options):
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    option_str = "\n".join([f"{l}. {o}" for l, o in zip(option_letters[:len(options)], options)])
    return option_str


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    source = data["source"]
    level = data["level"]
    if level == "hard" and source not in ("math", "rag"):
        is_retrieve = random.random() > (2 / 7)
    elif level == "very_hard" and source not in ("math", "rag"):
        is_retrieve = True
#    if source not in ("math", "rag"):
#        is_retrieve = True
#    elif source == "rag":
#        is_retrieve = True
#    elif source == "math":
#        is_retrieve = "level" in ["hard", "very_hard"]
    else:
        is_retrieve = False
    if apply_chat_template:
        question = data["question"]
        idx = data["idx"]
        """You are a helpful assistant.
        Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
        The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
        During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must #involve only a single triple**.
        Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""

        """You are an expert in solving multiple-choice questions and must analyze each option carefully and logically.
        You must reason step by step, explicitly considering the meaning and implications of *each* option, eliminating incorrect choices with clear justification, and identifying the best answer through careful comparison.
        The reasoning process and final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively:
        "<think> [Detailed reasoning here, including analysis of each option: 'Option A... Option B... etc.'] </think>\n\n<answer> [Final answer: A, B, C, ...] </answer>"""        
        """You are solving a multiple-choice question. Think step by step and reason carefully before answering.

During your reasoning, if you’re unsure about any fact, you may issue a **search query** like this:
`<|begin_of_query|> your concise query <|end_of_query|>`

* You can issue **multiple queries** at different steps in your reasoning.
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

Once documents are returned in this format:
`<|begin_of_documents|> ... <|end_of_documents|>`
use them to refine your reasoning before proceeding.

Use this output format:

```
<think> Step-by-step reasoning, including option analysis and retrieved information. </think>
<answer> Final answer. Use only the letter (A, B, C, etc.). </answer>
```
"""  
        """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

        During your reasoning, if you’re unsure about any fact, you may issue a **search query** like this:
        <|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

        * You can issue **multiple queries** at different steps in your reasoning.
        * **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

        Once documents are returned in this format:
        <|begin_of_documents|> ... <|end_of_documents|>

        use them to refine your reasoning before proceeding and continue reasoning immediately after reviewing the retrieved information.

        At the end of your reasoning, give your final answer in the following format:
        the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""
        """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

During your reasoning, if you're unsure about any fact, you can use the search tool by issuing a tool call in the following format:

<tool_call>
{"name": "search", "query": "your concise query (less than 20 words)"}
</tool_call>

A detailed explanation of this function is provided below:

{
  "type": "function",
  "function": {
    "name": "search",
    "description": "Search the web for information related to a specific factual question. You should use this tool to verify individual facts or resolve uncertainties during reasoning.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "A concise query (less than 20 words) to verify a specific fact"
        }
      },
      "required": ["query"]
    }
  }
}

**Tool usage instructions:**

* You can issue **multiple tool calls** during reasoning, whenever you encounter a factual uncertainty.
* Each tool call query must focus on **only one fact or statement**. Do **not** combine multiple ideas in a single query.

When documents are returned like this:
<|object_ref_start|>
... (search results here)
<|object_ref_end|>

review the documents and immediately update your reasoning based on them.

At the end of your reasoning, provide your final answer in this format:
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option).
"""
        """You are solving a factual open-domain question from a Knowledge Question Answering (KQA) task. The question requires step-by-step reasoning over real-world knowledge to identify a specific, factually correct answer.

Carefully analyze the question to understand the key entities, relationships, and constraints involved. Retrieve and consider relevant factual knowledge, and reason logically to identify the most accurate answer. The reasoning process should include detailed considerations such as analyzing the question, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps when needed.

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<search> your concise query (less than 20 words) </search>

* You can issue **multiple queries** at different steps in your reasoning.  
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.

Once documents are returned in this format:  
<information>  
... (search results here)  
</information>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the question until you're confident in the answer.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
        """You are a helpful assistant.
        Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
        The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
        During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
        Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""
        if is_retrieve and data['source'] not in ['math', 'rag']:
            sys_prompt = """You are solving a multiple-choice question. Analyze each option carefully and logically. Think step by step: consider the meaning and implications of each option, eliminate incorrect ones with clear reasoning, and select the best answer through comparison.

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<search> your concise query (less than 20 words) </search>

* You can issue **multiple queries** at different steps in your reasoning.  
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.
  * ✅ Example:
    * <search> What are the common symptoms of pneumonia? </search>
    * <search> What is the typical treatment for pneumonia in elderly patients? </search>
  * ❌ Do **not** combine them like this:
    * <search> What are the symptoms and treatments for pneumonia in elderly patients? </search>
* You may issue **at most four queries** in total — use them wisely.

Once documents are returned in this format:  
<information>
... (search results here)  
</information>

use the retrieved documents to verify, reject, or revise your prior reasoning about the options. Then continue analyzing the options until you're confident in your answer.

At the end of your reasoning, give your final answer in the following format:  
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""
            option_str = get_option_str(data["options"])
            Question = f"Question: {question}\nOptions:\n{option_str}"
            messages_chat=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": Question}
            ]
            prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        elif data['source'] != 'rag' and data['source'] != 'math':
            sys_prompt = """You are solving a multiple-choice question. Think step by step and use careful reasoning.

For each question, **analyze all options one by one**. For each option:
- Consider its meaning and implications.
- Evaluate whether it is correct or incorrect, and **explain why**.
- Eliminate incorrect options with clear, logical reasoning.

After analyzing all options, compare the remaining ones and choose the best answer.

At the end of your reasoning, give your final answer in the following format:  
the correct answer is: A, B, C, D, etc. (only the letter corresponding to the correct option)."""
#\\boxed{FinalAnswerLetter}  (e.g., \\boxed{C}).
            option_str = get_option_str(data["options"])
            Question = f"Question: {question}\nOptions:\n{option_str}"
            messages_chat=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": Question}
            ]
            prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        elif data['source'] == "math" and is_retrieve == False:
            sys_prompt = """You are sovling a math problem. Think step by step to solve it. 

The reasoning process includs detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
            Question = f"Question: {question}"
            messages_chat=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": Question}
            ]
            prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        elif data['source'] == "math" and is_retrieve == True:
            sys_prompt = """You are solving a math problem. Think step by step to solve it.

The reasoning process includes detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.

During your reasoning, if you're unsure about a factual concept — such as a definition, formula, theorem, or mathematical constant — you may issue a **search query** to clarify it.

Format your query using the following template (each query must target only one fact):

<|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

✅ Examples:
<|begin_of_query|> Definition of Möbius function <|end_of_query|>  
<|begin_of_query|> Formula for variance of Bernoulli distribution <|end_of_query|>

❌ Do NOT query for reasoning-related content like:
- Whether a solution approach is valid
- How to compute a specific value
- Multi-step deductions or conclusions

You may issue at most **four** search queries per problem — use them wisely.

When documents are returned in this format:
<|begin_of_documents|>  
... (search results here)  
<|end_of_documents|>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the question until you're confident in the answer.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
            Question = f"Question: {question}"
            messages_chat=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": Question}
            ]
            prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
        else:
            sys_prompt = """You are solving a factual open-domain question from a Knowledge Question Answering (KQA) task. The question requires step-by-step reasoning over real-world knowledge to identify a specific, factually correct answer.

Carefully analyze the question to understand the key entities, relationships, and constraints involved. Retrieve and consider relevant factual knowledge, and reason logically to identify the most accurate answer. 

During your reasoning, if you're unsure about any fact, you may issue a **search query** like this:  
<|begin_of_query|> your concise query (less than 20 words) <|end_of_query|>

* You can issue **multiple queries** at different steps in your reasoning.
* **Each query must target only one fact or statement**. Do not combine multiple ideas in a single query.
  * ✅ Example:
    * <|begin_of_query|> When did Einstein move to the United States? <|end_of_query|>
    * <|begin_of_query|> Why did Einstein leave Germany? <|end_of_query|>
  * ❌ Do **not** combine them like this:
    * <|begin_of_query|> When did Einstein move to the US and why did he leave Germany? <|end_of_query|>
* You may issue **at most five queries** in total — use them wisely.
Once documents are returned in this format:  
<|begin_of_documents|> 
... (search results here)  
<|end_of_documents|>

Use the evidence to confirm or revise your reasoning. Then continue analyzing the question until you're confident in the answer.

At the end of your reasoning, give your final answer in the following format:  
\\boxed{YOUR_ANSWER}"""
            Question = f"{question}"
            messages_chat=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": Question}
            ]
            prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
    else:

        """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

        """---
The User asks a question, which belongs to either a **math** or a **retrieval (RAG)** task.  
The Assistant is required to solve it step by step, thinking carefully before answering.

The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly.  
This reasoning should be enclosed within `<think>` and `</think>` tags.  
Then, it provides the **final concise answer** within `<answer>` and `</answer>` tags.

If the question is about **mathematics**, the Assistant is strictly **forbidden to use search**, and must rely solely on its internal reasoning ability.

If the question is about **retrieval (RAG)**, the Assistant **should search if unsure** during the thinking phase using the following format:

<|begin\_of\_query|> keyword\_1 keyword\_2 keyword\_3 <|end\_of\_query|>

A search query must involve **only a single triple**.

The retrieved results will then be provided within:

<|begin\_of\_documents|> ...search results... <|end\_of\_documents|>

Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.
---

Output Format:

<think>
Step-by-step reasoning process goes here. Explain logic clearly and slowly.
Use search query only if necessary and allowed (i.e., for RAG only).
</think>

<answer>
Final concise answer goes here.
</answer>

User:{question}
Assistant: <think>"""#0.1.3
        """The User asks a question, and the Assistant solves it.
The Assistant is required to solve it step by step, thinking carefully before answering.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".
The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.  The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

User:{question}
Assistant: <think>"""
        """The User asks a question, and the Assistant solves it.
        The Assistant is required to solve it step by step, thinking carefully before answering.
        The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> If the question is multiple-choice, only output the corresponding uppercase letter (e.g., A, B, C, D). If it is not multiple-choice, output the complete answer normally. </answer>".
        During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
        Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".
        The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.  The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

        User: {question}
        Options:\n{option_str}
        Assistant: <think>"""#mmlu0.2
        """The User asks a question, and the Assistant solves it.  
The Assistant is required to solve it step by step, thinking carefully before answering.  
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,  
"<think> reasoning process here </think>\n\n<answer> If the question is multiple-choice, only output the corresponding uppercase letter (e.g., A, B, C, D). If it is not multiple-choice, output the complete answer normally. </answer>".

During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary, using the format:  
"<|begin_of_query|> keyword_1 keyword_2 ... <|end_of_query|>".  

**Each query must correspond to only ONE triple or factual statement.**  
Avoid comma-separated lists or conjunctions like "and"/"or" that include multiple ideas.  
If multiple pieces of information are needed, issue multiple separate queries.

Then, the search system will provide the Assistant with retrieval information in the format:  
"<|begin_of_documents|> ...search results... <|end_of_documents|>".  

The Assistant must **first think through the reasoning process in detail**, explaining each logical step clearly. Be sure to explain how retrieved documents help with solving the question during the `<think>` stage.  
The Assistant does not retrieve information lightly (e.g., for math problems) unless the situation truly requires external knowledge, such as in knowledge-based question answering.

User: {question}  
Options:\n{option_str}  
Assistant: <think>"""#mmlu0.3及以后
        base_prompt = """The User asks a question, and the Assistant solves it.
        The Assistant is required to solve it step by step, thinking carefully before answering.
        The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> If the question is multiple-choice, only output the corresponding uppercase letter (e.g., A, B, C, D). If it is not multiple-choice, output the complete answer normally. </answer>".
        User: {question}
        Options:\n{option_str}
        Assistant: <think>""" #mmlu直接回答baseline_prompt v0.1.0
        base_prompt = """The User asks a question, and the Assistant solves it. The Assistant is an expert in solving multiple-choice questions and must analyze each option carefully and logically.

The Assistant must reason step by step, explicitly considering the meaning and implications of *each* option, eliminating incorrect choices with clear justification, and identifying the best answer through careful comparison.

The reasoning process and final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively:
"<think> [Detailed reasoning here, including analysis of each option: 'Option A... Option B... etc.'] </think>\n\n<answer> [Final answer: A, B, C, ...] </answer>"

User: {question}
Options:\n{option_str}
Assistant: <think>""" #mmlu直接回答baseline_prompt v0.1.1
        question = data["question"]
        option_str = get_option_str(data["options"])
#        prompt = (
#            "Please answer the following math question. Think step by step to solve it.\n\n"
#            "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
#            f"Question:\n{question}\n\n"
#            "Solution:"
#        )
        
        idx = data["idx"]
        prompt = base_prompt.format(question=question, option_str=option_str)
        #prompt = base_prompt.format(question=question)
    return str(idx) + "<|idx_prompt_split|>" + prompt + "<|source_prompt_split|>" + data["source"] + "<|retrieve_prompt_split|>" + str(is_retrieve)


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)
        # print("len(self.prompts):",len(self.prompts))
        # print("self.prompts[0:5]:",self.prompts[0:5])
        # kill

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
