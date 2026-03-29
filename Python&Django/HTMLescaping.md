# HTML 이스케이프

html 이스케이프 : HTML에서 특수문자를 “문자 그대로 보이게” 바꾸는 것 

Django 템플릿은 자동으로 html 이스케이프 처리를 수행함.




Django 내장 템플릿 태그

1. autoescape
- 블럭 내에서 자동적으로 모든 이스케이프를 수행할 지 결정.
- 시작 태그는 {% autoescape on %}혹은 {% autoescape off %} 사용. 
- 종료 태그 endautoescape 사용
- safe 필터가 적용된 변수는 autoescape의 예외다.

2. escape
- 텍스트의 HTML을 이스케이프 처리한다.
- '내장 필터'의 일종.
- 위의 autoescape로 자동 이스케이핑이 적용되는 변수에 이 함수를 적용하면 이스케이핑이 한 번만 수행된다. 따라서 자동 이스케이핑이 적용되는 환경에서도 안전하게 사용할 수 있음. 여러 번의 이스케이핑을 적용하려면 force_escape필터를 사용.

> 이스케이프 코드
>
> &lt; is converted to &amp;lt;  
>
> &gt; is converted to &amp;gt;  
>
> ' (single quote) is converted to &amp;#x27;  
>
> " (double quote) is converted to &amp;quot;  
>
> &amp; is converted to &amp;amp;


3. safe 
- 출력 전에 '추가적인 HTML 이스케이핑이 필요하지 않은 문자열'로 표시. 
- 자동 이스케이핑이 꺼져 있으면 효과 없음.


*출처 : https://docs.djangoproject.com/ko/6.0/ref/templates/builtins/*



기본적으로 Django는 autoescape on 상태이기 때문에, 우리는 safe 사용.

