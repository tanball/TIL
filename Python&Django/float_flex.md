# Float와 Flex

부트스트랩에서 물체를 배치하는 방법들.

## Float
CSS속성.
Bootstrap 4에서 사용되던 방식. flex로 대체를 권장.
- float-start
- float-end
- float-none
위 세가지를 사용해 왼쪽/오른쪽으로 이동시키거나, 이동을 막는다.
그리드 시스템과 같은 브레이크포인트를 사용
단, Flex요소에는 영향을 미치지 않는다.
현재 뷰포트 크기에 따라 플로팅

> These utility classes float an element to the left or right, or disable floating, based on the current viewport size using the CSS float property. !important is included to avoid specificity issues. These use the same viewport breakpoints as our grid system. Please be aware float utilities have no effect on flex items.
>
> 출처 : https://getbootstrap.com/docs/5.0/utilities/float/


## Flex
부트스트랩5의 핵심 레이아웃 방식
그리드 열, 탐색 메뉴, 구성 요소 등의 레이아웃, 정렬 및 크기 관리.
- d-flex (flexbox 컨테이너)
- justify-content-* (가로정렬)
- align-items-* (세로정렬)
- ms-auto, me-auto (자동 여백으로 밀기)
가장 유연하며, 거의 모든 ui를 구현 가능하다. 공식적으로 권장된다.

사용법
1. 배치할 요소가 존재하는 컨테이너에 d-flex를 준다.(flex컨테이너 생성)
2. 컨테이너에 정렬 클래스를 추가하여 하위 요소들의 배치를 지정한다.
3. ms-auto등은 하위 요소에 배치될 수도 있는데, 그에 따라 아이템 하나만 밀 수도 있다.

float와 달리, 움직일 요소가 아니라 그 요소가 포함될 컨테이너에서 하위 요소 전체의 배치를 지정.

> Quickly manage the layout, alignment, and sizing of grid columns, navigation, components, and more with a full suite of responsive flexbox utilities. For more complex implementations, custom CSS may be necessary.
>
> 출처 : https://getbootstrap.com/docs/5.0/utilities/flex/


## 그 외
1. Grid
행-열 구조의 2차원 레이아웃으로 사용
- row
- col-*
- g-* (gap)

이렇게 분리한 행렬 한 칸 한칸에 사용.

카드 배치, 목록 등.

2. Utilities
간단한 정렬용.
- text-end (텍스트/인라인요소 우측정렬)
- mx-auto (가운데 정렬)
- w-100 (너비 100%)
- d-block, d-inline ...

빠르고 간단.

복잡한 구조에는 한계가 있다.

3. Position
절대 위치 제어.
정확한 위치 지정에 사용
- position-relative
- position-absolute
- top-0, end-0 등

오버레이/닫기 버튼 등에 사용

일반 레이아웃에 부적합