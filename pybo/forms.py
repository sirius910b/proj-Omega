"""
폼(Form)은 페이지 요청시 전달되는 파라미터들을
쉽게 관리하기 위해 사용하는 클래스.
폼은 필수 파라미터의 값이 누락되지 않았는지,
파라미터의 형식은 적절한지 등을 검증할 목적으로 사용한다!

*forms.Form 과 forms.ModelForm 의 차이*
모델 폼은 모델과 연결된 폼으로 폼을 저장하면 연결된 모델의 데이터를
저장할 수 있는 폼이다!
이너 클래스인 Meta 클래스가 반드시 필요함
-> Meta 클래스에는 사용할 모델과 모델의 속성을 적어야 한다.
"""
from django import forms
from pybo.models import Question, Answer


# forms.Form 과 forms.ModelForm 의 차이는
# 모델 폼은 모델과 연결된 폼으로 폼을 저장하면 연결된 모델의 데이터를
# 저장할 수 있는 폼이다! -> 이너 클래스인 Meta 클래스가 반드시 필요함
# Meta 클래스에는 사용할 모델과 모델의 속성을 적어야 한다.
class QuestionForm(forms.ModelForm):
    class Meta:
        # 사용할 모델
        model = Question
        # QuestionForm 에서 사용할 Question 모델의 속성
        fields = ['subject', 'content']

        labels = {
            'subject': '제목',
            'content': '내용',
        }


class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }