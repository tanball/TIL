# django-crispy-forms 

django의 기본 폼을 더 쉽게 예쁘게 렌더링하게 도와주는 라이브러리.


'''
pip install django-crispy-forms
pip install crispy-bootstrap5
'''

```
INSTALLED_APPS = (
    ...
    "crispy_forms",
    "crispy_bootstrap5",
    ...
)

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"

CRISPY_TEMPLATE_PACK = "bootstrap5"
```

버전 2.0으로 넘어오며 부트스트랩5가 외부 패키지(서드파티 패키지)로 분리되었으며, 그 사용법에도 변동이 있음.

> 참고 
>
> newrelease 사이트의 2.0 릴리즈 안내
>
> https://newreleases.io/project/pypi/django-crispy-forms/release/2.0a1?utm_source=chatgpt.com
>
> 부트스트랩 5 외부 패키지 설명
>
> https://github.com/django-crispy-forms/crispy-bootstrap5
>
> 문서 내 README.md 파일 참고.


*newrelease.io 사이트 : 각종 소프트웨어의 릴리즈 알림을 제공하고, 릴리즈 시 변화에 대한 안내를 제공.*
*내 프로젝트에 사용하는 라이브러리를 알림 등록. 프로젝트 진행 중 릴리즈로 인한 문제를 막는데 도움이 됨.*