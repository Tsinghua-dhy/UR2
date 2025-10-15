# import requests
# import json
# # print(len(a))
# # print(len(a['query']))
# # exit()
# # 设置 API 的 URL
# url = "http://0.0.0.0:1278/get_reward"  # 这里假设你的 FastAPI 服务运行在本地的 5001 端口


# def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
#     """Synchronous request API wrapper"""
#     headers = {
#         "Content-Type": "application/json",
#     }
#     for _ in range(try_max_times):
#         try:
#             response = requests.post(url=url, json=data, headers=headers, timeout=360)
#             response.raise_for_status()  # Raise an HTTPError for bad responses
#             response = response.json()
#             assert score_key in response, f"{score_key} not in {response}"
#             return response.get(score_key)
#         except requests.RequestException as e:
#             print(f"Request error, please check: {e}")
#         except Exception as e:
#             print(f"Unexpected error, please check: {e}")
#         # time.sleep(1)

#     raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")

# scores = request_api_wrapper(url, a, "rewards")
# print(scores)

# # 创建请求数据（问题列表）
# # data=a
# # data = {
# #     "query": [
# #         "What is 2+2?",
# #         "How does gravity work?",
# #         "1+1=?"
# #     ]
# # }
# # data =a
# # print(data)
# # # 发送 POST 请求到 API
# # response = requests.post(url, json=data)

# # # 打印返回的结果
# # if response.status_code == 200:
# #     result = response.json()  # 解析 JSON 响应
# #     print("Received rewards:", result["rewards"])
# # else:
# #     print(f"Error: {response.status_code}, {response.text}")
query = '<|im_start|>system\nYou are skilled at solving problems through step-by-step reasoning, leveraging both your own knowledge and external search engine.<|im_end|>\n<|im_start|>user\nGiven a complex multi-hop question, you need to reason step by step and **enclose the final answer within `<answer></answer>` tags**.\nYou have the ability to perform web searches. For uncertain knowledge, you **can utilize the external Search Engine to retrieve knowledge for solving questions**. You need to **provide the search query (only keywords) enclosed in the `<query></query>` tag**, then you will receive the relevant documents enclosed in the <tool_call></tool_call> tags.\n\nThe reasoning process should include detailed considerations such as analyzing questions, decomposing the questions, performing searching, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.\n\nDuring this process, you should use casual, genuine phrases such as: "Hmm...", "Wait, let me think about...", "Actually...", "Aha...", "Now that I look at it...", "This reminds me of...", "I wonder if...", "But then again...", "Let\'s see if...", "Alternatively...", "Let\'s summarize existing information...", etc., to make the reasoning process coherent, clear, and logically sound, effectively simulating human cognitive processes.\n\n**Guidelines**:\n- You should **show the reasoning process**, and **ONLY return the FINAL ANSWER within `<answer></answer>` tags**. For example: The Reasoning Process...<answer>The Final Answer</answer>.\n- When you need to retrieve external knowledge, you should **provide the search query (only keywords) enclosed in the `<query></query>` tag**. For example: <query>Query consist of Keywords</query>.\n- When done searching, continue your reasoning.\n\n[Question]\nWhat was the name of the woman born in 1943 who had an affair with John F. Kennedy toward the end of his life?\n<|im_end|>\n<|im_start|>assistant\nTo determine the name of the woman who had an affair with John F. Kennedy near the end of his life, let\'s proceed step by step.\n\nFirst, we need to consider who John F. Kennedy was and when he was alive. Kennedy was a prominent American politician who served as the 35th President of the United States. He was assassinated in Dallas, Texas, on November 22, 1963, at the age of 46. This suggests the affair likely occurred in the later 1960s or early 1970s.\n\nNext, we recall that Kennedy had several affairs during his life, but we need to identify a specific one near the end of his life. One well-known affair is his affair with Marilyn Monroe.\n\nTo confirm the identity of the woman, we can search for information about Marilyn Monroe in the 1960s and her relationship with Kennedy.\n\n**Search Query:** Marilyn Monroe 1960s affair with John F. Kennedy\n\nNow, upon reviewing the search results, we can identify that Marilyn Monroe had an affair with John F. Kennedy in the late 1960s, coinciding with his presidency. Monroe was famously spotted with Kennedy at the 1962 Kennedy Center Honors Awards ceremony and again at the 1964 first inauguration, where she had her first kiss with him.\n\nTherefore, based on this information, the answer to the question is: Marilyn Monroe<|im_end|>'


question_prompt = query.split("<|im_end|>\n<|im_start|>user")[-1].split("<|im_end|>\n<|im_start|>assistant")[0].strip()
question = question_prompt.split("[Question]\n")[-1].strip()

solution = query.split("<|im_end|>\n<|im_start|>assistant")[-1].strip()
print(list[question])
print(list[solution])