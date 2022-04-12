from transformers import TFBertForMaskedLM, BertForMaskedLM
from transformers import AutoTokenizer
from transformers import FillMaskPipeline

model = BertForMaskedLM.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

inputs = tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='pt')

print(inputs['input_ids']) # input 정수 인코딩
print(inputs['token_type_ids']) # 세그먼트 인코딩
print(inputs['attention_mask']) # 패딩 결과

pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
result = pip('축구는 정말 재미있는 [MASK]다.')
print(result)
result = pip('어벤져스는 정말 재미있는 [MASK]다.')
print(result)