# Bleach

허용 목록 기반의 HTML 정제 라이브러리.

기능
1. 마크업과 속성을 이스케이프 처리하거나 제거한다.
2. 신뢰할 수 없는 출처의 텍스트 검증
3. html5 라이브러리 기반. 불균형, 잘못 중첩된 태그 등 수정 가능.

```
pip install bleach
```

```
import bleach

bleach.clean('an <script>evil()</script> example')
u'an &lt;script&gt;evil()&lt;/script&gt; example'

bleach.linkify('an http://example.com url')
u'an <a href="http://example.com" rel="nofollow">http://example.com</a> url'
```


> Sanitize
>
> bleach.clean()
>
> html 컨텍스트에서 사용하기 위해 html 조각을 정제하는 데 사용됨.
>
> 텍스트 조각 분석. 태그, 속성 및 기타 요소 정제
>
> 이스케이프되지 않은 문자, 닫히지 않은 태그, 잘못 중첩된 태그 등도 처리
>
> html속성, css, js, js템플릿, JSON, xhtml, SVG 또는 기타 컨텍스트에서 사용하면 안됨.
>
> 이 출력값을 html 속성값으로 사용하려면 jinja나 django의 이스케이프 함수를 거쳐야 함.





> 참고
>
> https://bleach.readthedocs.io/en/latest/
> 
> bleach 공식 문서