# 요청 방식 : GET과 POST


## GET 요청

**서버에서 데이터를 조회할 때 쓰는 http 요청 방식**

유저가 서버의 데이터를 조회할 때는 get 방식의 요청을 한다.

유저가 서버에 데이터 조회를 요청하고, 서버는 요청받은 데이터를 캐싱하여 url에 담아 보낸다.

데이터가 url에 포함하여 보낼 수 있는 길이로 제한된다.

>The GET method requests transfer of a current selected representation for the target resource. GET is the primary mechanism of information retrieval and the focus of almost all performance optimizations. Hence, when people speak of retrieving some identifiable information via HTTP, they are generally referring to making a GET request.
>
>It is tempting to think of resource identifiers as remote file system pathnames and of representations as being a copy of the contents of such files. In fact, that is how many resources are implemented (see Section 9.1 for related security considerations). However, there are no such limitations in practice. The HTTP interface for a resource is just as likely to be implemented as a tree of content objects, a programmatic view on various database records, or a gateway to other information systems. Even when the URI mapping mechanism is tied to a file system, an origin server might be configured to execute the files with the request as input and send the output as the representation rather than transfer the files directly. Regardless, only the origin server needs to know how each of its resource identifiers corresponds to an implementation and how each implementation manages to select and send a current representation of the target resource in a response to GET.
>
>A client can alter the semantics of GET to be a "range request", requesting transfer of only some part(s) of the selected representation, by sending a Range header field in the request ([RFC7233]).
>
>A payload within a GET request message has no defined semantics; sending a payload body on a GET request might cause some existing implementations to reject the request.
>
>The response to a GET request is cacheable; a cache MAY use it to satisfy subsequent GET and HEAD requests unless otherwise indicated by the Cache-Control header field (Section 5.2 of [RFC7234]).
>
>GET 메서드는 대상 리소스(target resource)에 대해 **현재 선택된 표현(current selected representation)**을 전송하도록 요청하는 메서드이다.
>
>GET은 정보를 검색하는 주요 메커니즘이며, 거의 모든 성능 최적화의 중심이 된다.
>
> *출처 : RFC7231문서(https://httpwg.org/specs/rfc7231.html?utm_source=chatgpt.com#rfc.section.4.3.1)*


## POST 요청

**서버에 데이터를 보내서 생성/변경 작업을 할 때 쓰는 http 요청 방식**

유저가 서버에 데이터를 보내 이 데이터를 저장하라고 요청한다.

데이터는 http body에 포함되어 전송되며, 따라서 길이 제한은 없고 캐싱도 불필요하다.

>The POST method requests that the target resource process the representation enclosed in the request according to the resource's own specific semantics. For example, POST is used for the following functions (among others):
>
>Providing a block of data, such as the fields entered into an HTML form, to a data-handling process;
>
>Posting a message to a bulletin board, newsgroup, mailing list, blog, or similar group of articles;
>
>Creating a new resource that has yet to be identified by the origin server; and
>
>Appending data to a resource's existing representation(s).
>
>An origin server indicates response semantics by choosing an appropriate status code depending on the result of processing the POST request; almost all of the status codes defined by this specification might be received in a response to POST (the exceptions being 206 (Partial Content), 304 (Not Modified), and 416 (Range Not Satisfiable)).
>
>If one or more resources has been created on the origin server as a result of successfully processing a POST request, the origin server SHOULD send a 201 (Created) response containing a Location header field that provides an identifier for the primary resource created (Section 7.1.2) and a representation that describes the status of the request while referring to the new resource(s).
>
>Responses to POST requests are only cacheable when they include explicit freshness information (see Section 4.2.1 of [RFC7234]). However, POST caching is not widely implemented. For cases where an origin server wishes the client to be able to cache the result of a POST in a way that can be reused by a later GET, the origin server MAY send a 200 (OK) response containing the result and a Content-Location header field that has the same value as the POST's effective request URI (Section 3.1.4.2).
>
>If the result of processing a POST would be equivalent to a representation of an existing resource, an origin server MAY redirect the user agent to that resource by sending a 303 (See Other) response with the existing resource's identifier in the Location field. This has the benefits of providing the user agent a resource identifier and transferring the representation via a method more amenable to shared caching, though at the cost of an extra request if the user agent does not already have the representation cached.
>
>POST 메서드는 요청에 포함된 표현(representation)을 대상 리소스(target resource)가 자신의 고유한 의미론(semantics)에 따라 처리하도록 요청한다.
>
>예를 들어 POST는 다음과 같은 기능에 사용된다 (이 외에도 다양한 용도가 있다).
>
> - HTML 폼에 입력된 필드와 같은 데이터 블록을 데이터 처리 프로세스에 제공
> - 게시판, 뉴스그룹, 메일링 리스트, 블로그 또는 유사한 글 모음에 메시지를 게시
> - 원본 서버(origin server)에 의해 아직 식별되지 않은 새로운 리소스를 생성
> - 기존 리소스의 표현(representation)에 데이터를 추가
>
> 원본 서버는 POST 요청 처리 결과에 따라 적절한 상태 코드(status code)를 선택함으로써 응답의 의미를 나타낸다.
>
>이 명세에서 정의된 거의 모든 상태 코드는 POST 요청에 대한 응답으로 받을 수 있다.
>
>*출처 : RFC7231문서(https://httpwg.org/specs/rfc7231.html?utm_source=chatgpt.com#rfc.section.4.3.1)*


### dispatch() 메서드

**일반적으로, 요청의 종류가 get인지 post인지 식별 후 그에 맞는 처리를 수행**
