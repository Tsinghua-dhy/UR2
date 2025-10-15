import json

# 输入和输出文件路径
input_file_path = '/AIRPFS/lwt/corpus/kilt_knowledgesource.jsonl'
output_file_path = '/AIRPFS/lwt/corpus/wiki_kilt_100_really.tsv'

# 初始化索引
idx = 0
line_num=0
# 打开输出文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 写入TSV表头
    # output_file.write("idx\ttext\n")

    # 读取输入的 JSONL 文件
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line_num+=1
            if line_num% 100000==0:
                print("Finished Entity:",line_num)
                print("Now 100-words paragraph:",idx)
                print("=="*20)
            data = json.loads(line)
            # 假设 text 是一个列表
            text_list = data.get("text", [])
            title=data["wikipedia_title"]
            # 将 text 列表连接成一个字符串
            full_text = "".join(text_list)
            full_text = full_text.replace("BULLET::::","").replace("Section::::","")
            num_1=full_text.count("::::")
            num_2=full_text.count("print(a.split())")
            num_3=full_text.count("Section::::")
            if num_1!=num_2+num_3:
                print(full_text)
                print("=="*10)
            # assert num_1==num_2+num_3
            # if "::::" in full_text:
            #     print(full_text)
            #     # kill

            # 按空格拆分成单词
            words = full_text.split()

            # 每 100 个单词作为一个段落
            for i in range(0, len(words), 100):
                paragraph = " ".join(words[i:i + 100])
                paragraph = f"{idx}\t{title}   {paragraph}"
                output_file.write(f"{paragraph}\n")
                idx += 1

print(f"处理完成，结果已写入 {output_file_path}。")
# 360,000,000