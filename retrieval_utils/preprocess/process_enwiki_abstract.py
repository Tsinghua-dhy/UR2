import bz2
import json
import glob

output_file = 'enwiki_2017_abstract.tsv'
input_files = glob.glob('/AIRPFS/lwt/corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts/**/*.bz2', recursive=True)  # 递归查找

with open(output_file, 'w', encoding='utf-8') as out_f:
    for bz2_file in input_files:
        with bz2.open(bz2_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    title = data.get("title", "").replace("\t", " ").replace("\n", " ").strip()
                    text_list = data.get("text", [])
                    text_list = [text.strip() for text in text_list]
                    text = ' '.join(text_list).replace("\t", " ").replace("\n", " ").strip()
                    if title and text:
                        combined = f"{title}   {text}\n"
                        out_f.write(combined)
                except Exception as e:
                    print(f"跳过出错行 in {bz2_file}: {e}")
