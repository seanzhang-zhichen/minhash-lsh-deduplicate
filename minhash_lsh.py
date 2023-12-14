import os
import time
import json
import jieba
import joblib
import jsonlines
import multiprocessing
import pandas as pd
from loguru import logger
from tqdm import tqdm
from multiprocessing import Pool
from datasketch import MinHash, MinHashLSH

class JiebaSegment(object):
    def __init__(self) -> None:
        pass

    def segment(self, text):
        res = list(jieba.cut(text))
        return res

    def segment_multi_p(self, text_list, core_num=None):
        if core_num is None:
            core_num = multiprocessing.cpu_count()
        logger.info(f"开启 {core_num} 个进程分词")

        pool = Pool(core_num)
        token_result = []
        for result in tqdm(pool.imap(func=self.segment, iterable=text_list), total=len(text_list), desc="分词"):
            token_result.append(result)
        return token_result


class MinhashLshDeduplicate:
    def __init__(self) -> None:
        data_path = "你的数据集路径"
        with open(data_path, 'r', encoding="utf-8") as f:
            data_list = json.load(f)
        self.data_list = [item["title"] for item in data_list]
        
        self.jieba_segment = JiebaSegment()

    # 创建一个 MinHash 对象
    def create_minhash(self, word_list):
        minhash = MinHash(num_perm=128)  # num_perm 是哈希函数的数量，可以根据需要调整
        # for word in word_list:
        #     minhash.update(word.encode('utf8'))

        # update_batch 比 update 快一些
        minhash.update_batch([w.encode('utf8') for w in word_list])

        return minhash

    def deduplicate(self):
        # 创建一些示例数据（中文长句子）
        sentences = self.data_list[:10000]

        # 创建 MinHash 对象并插入到 LSH 中
        lsh = MinHashLSH(threshold=0.5, num_perm=128)  # threshold 是相似度阈值，可以根据需要调整

        t0 = time.time()
        for idx, sentence in enumerate(sentences):
            minhash = self.create_minhash(self.jieba_segment.segment(sentence))
            lsh.insert(idx, minhash)
        t1 = time.time()
        print(f"创建minhash对象共耗时: {t1 - t0} s")

            
        query_sentence = "踝部骨折患者的治疗及护理措施研究与分析"

        
        # 查找相似的集合
        query_minhash = self.create_minhash(self.jieba_segment.segment(query_sentence))
        idxs = lsh.query(query_minhash)
        
        # 输出相似度分数
        for idx in idxs:
            minhash = self.create_minhash(self.jieba_segment.segment(sentences[idx]))
            jaccard_similarity = query_minhash.jaccard(minhash)
            print(f"与 {query_sentence} 相似的句子 {sentences[idx]} 的相似度分数为: {jaccard_similarity}")
            # output
            # 与 sentence 相似的句子 8 的相似度分数为: 0.8046875
        t2 = time.time()
        print(f"去重耗时: {t2 - t1} s")
            
    
if __name__ == "__main__":
    minhash_lsh = MinhashLshDeduplicate()
    minhash_lsh.deduplicate()
