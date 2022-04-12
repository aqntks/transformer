from transformers import TFBertForNextSentencePrediction, BertForNextSentencePrediction
from transformers import AutoTokenizer
import torch.nn as nn
import torch

model = BertForNextSentencePrediction.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 이어지는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = nn.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', torch.argmax(probs).numpy())

# 상관없는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "극장가서 로맨스 영화를 보고싶어요"
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = nn.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', torch.argmax(probs).numpy())