"""
모델을 수정한 이후엔 반드시 아래 코드를 셸에 작성할 것!!
python manage.py makemigrations

Select an option에서 1은 admin 계정의 id 값을 의미함
"""

from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Question(models.Model):
    # 계정이 삭제되면 이 계정이 작성한 질문을 모두 삭제
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question')
    # 제목처럼 글자수의 길이가 제한적인 경우엔 CharField 사용
    subject = models.CharField(max_length=200)
    # 내용처럼 글자수의 길이가 제한적이지 않은 경우엔 TextField 사용
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User, related_name='voter_question') # 추천인 추가

    def __str__(self):
        return self.subject


class Answer(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_answer')
    # models.CASCADE 는 질문이 삭제될 경우 답변도 함께 삭제함
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User, related_name='voter_answer')

    def __str__(self):
        return self.content
