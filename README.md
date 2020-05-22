# Light weight model of styling chatbot
가벼운 모델을 웹호스팅하기 위한 레포지토리입니다.\
원본 레포지토리는 다음과 같습니다.  [바로 가기](https://github.com/km19809/Styling-Chatbot-with-Transformer)

## 요구사항

이하의 내용은 개발 중 변경될 수 있으니 requirements.txt를 참고 바랍니다.
```
torch~=1.4.0
Flask~=1.1.2
torchtext~=0.6.0
hgtk~=0.1.3
konlpy~=0.5.2
chatspace~=1.0.1
```

## 사용법
`light_chatbot.py [--train] [--per_soft|--per_rough]`

* train: 학습해 모델을 만들 경우에 사용합니다. \
사용하지 않으면 모델을 불러와 시험 합니다.
* per_soft: soft 말투를 학습 또는 시험합니다.\
per_rough를 쓴 경우 rough 말투를 학습 또는 시험합니다.\
두 옵션은 양립 불가능합니다.

`app.py`

챗봇을 시험하기 위한 간단한 플라스크 서버입니다.