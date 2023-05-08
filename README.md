# translation_example

## 데이터 프로세싱 코드
```
raw_data_preprocessing.ipynb  # 파일 확인
```

## Vanila Transformer 구현 모델
```
$ vanila_transformer_train.py  # 디버깅 해보면 쉽게 이해 가능
$ anila_transformer_inference.py  # greedy 한 방식으로 generation 구현
```

## mT5 모델 학습 코드
```
$ run_translation.py  # Huggingface Seq2SeqTrainer Argument 숙지 필요!
```

## shell file
```
$ ./sh_folder/mt5_small_train.sh
```

명령어 huggingface Trainer 기본 Arguments 들로 거의 구성되어 있어서 쉽게 파악 가능!
