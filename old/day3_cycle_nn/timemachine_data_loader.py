import collections
import random
import re
import torch
from d2l import torch as d2l

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()

tokens = d2l.tokenize(lines)

# tokens1 = d2l.tokenize(lines, 'char')
# for i in range(10):
#     print(tokens1[i])

vocab = d2l.Vocab(tokens)
print(vocab._token_freqs[:10])
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

print("=== tokens每个元素是一行token， ===")
corpus = [token for line in tokens for token in line]
print(corpus[:10])

print("=== corpus是一个单词， bigram_tokens是二连单词 ===")
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
print(bigram_tokens[:10])
vocab = d2l.Vocab(bigram_tokens)
print(vocab._token_freqs[:10])



