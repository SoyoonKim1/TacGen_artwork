# TacGen 프로토타입 기획서

## 1. 기획 배경과 제안서 해석

제안서의 그림 6, 9, 17은 다음과 같은 하나의 콘텐츠 파이프라인으로 연결된다.

- **그림 6, 제안서 11쪽:** 전문가가 제작한 촉각 그래픽과 원작을 비교해 촉각 표현 기준을 정하고, 주요 객체·윤곽·영역·공간관계를 추출한다. 촉각 탐색에 필요한 정보를 단순화해 촉각 그래픽을 만들고 작품 자료 및 검색 기능에 연결한다.
- **그림 9, 제안서 14쪽:** 이미지 구조를 촉각 인지에 적합한 영역으로 나누고, 각 영역에 R1, R2 등의 ID와 객체·의미 정보를 부여한다. 촉각 그래픽은 예시처럼 60×40 표시 격자에 맞춰 표현한다.
- **그림 17, 제안서 22쪽:** 작품 이미지와 공식 작품 정보, 영역별 의미 정보를 연결한다. 접촉 위치를 Region ID로 바꾸고, 해당 영역에 맞는 자료를 찾아 설명·질의응답을 제공하면서 촉각 영역을 강조한다.

현재 폴더에는 제안서와 `data/sunflower.jpeg`가 있다. 전문가가 제작한 촉각 그래픽, 정답 영역 분할 데이터, 공식 작품 설명 자료, 학습된 모델은 아직 제공되지 않았다. 따라서 이번 프로토타입은 우선 한 작품으로 분할·촉각화 가능성을 검토하고, 모델 출력을 편집·검수할 수 있는 구조를 만든다. 자동 생성 결과를 전문가 검수 없이 정답으로 간주하지 않는다.

## 2. 목표와 범위

### 목표

1. 작품 이미지에서 객체와 영역 후보를 자동으로 얻는다.
2. 원작 구도와 중요한 공간관계를 보존하면서 촉각 표시용 영역으로 단순화한다.
3. 영역 ID와 작품 자료를 연결해 위치 기반 설명·질의응답을 제공한다.
4. 자동 분할, 촉각 그래픽 생성, 전문가 수정 결과를 구분해 저장하고 평가할 수 있게 한다.

### 초기 범위

- 단일 작품 `sunflower.jpeg`로 SAM 3 기반 분할 가능성을 확인한다.
- 촉각 그래픽은 우선 **결정론적 윤곽·영역 렌더러**를 기준선으로 만든다.
- TactileNet은 연구 설계와 비교 실험의 참고 모델로 검토한다. 원본 구도와 영역 ID를 보존한다는 것이 확인되기 전까지 주 렌더러로 채택하지 않는다.
- 기존 파이프라인의 JSON 스키마, Semantic Chunking, Material-Sensory Map 및 위치 기반 QA 기능을 모델 입력·출력에 연결한다.

### 초기 범위에서 제외

- 한 장의 이미지로 촉각 그래픽 생성 모델을 학습하는 것
- 모델의 질감 추론을 실제 물성 측정값처럼 제시하는 것
- 전문가·시각장애인 평가 없이 자동 생성물이 촉각 접근성 기준을 충족한다고 주장하는 것

## 3. SAM 3 적용안: 객체·영역 후보 분할

SAM 3는 짧은 개념 문구나 시각 프롬프트를 사용해 해당 개념의 객체 마스크를 찾는 모델이다. `sunflower.jpeg`에서는 다음과 같은 프롬프트 묶음을 시험한다.

- `sunflower head` / 해바라기 꽃송이
- `flower vase` / 꽃병
- `green stem` / 줄기
- `green leaves` / 잎
- 필요한 경우 `background` / 배경

프롬프트마다 마스크·박스·점수와 원본 크기 좌표를 보존한다. 같은 꽃송이를 여러 프롬프트가 중복 검출하거나, 줄기와 잎이 겹치는 경우에는 중복 제거와 관계 정리가 필요하다. 반복되는 붓 터치와 꽃잎의 경계를 모두 독립 영역으로 나누기보다, 촉각 탐색 목적에 따라 꽃송이·꽃병·줄기·잎처럼 **의미 있고 구별 가능한 큰 영역**으로 묶는 것을 우선한다.

SAM 3는 범용 객체 분할 모델이지 촉각 그래픽 설계기가 아니다. 따라서 출력 마스크는 다음 단계의 검수 대상이다.

1. 후보 객체가 원작의 중요한 내용을 빠뜨리지 않았는지 확인한다.
2. 중복·누락·과도하게 작은 마스크를 수정하거나 제거한다.
3. 인접하거나 같은 의미인 부분을 하나의 촉각 영역으로 병합한다.
4. 최종 영역에 안정적인 Region ID를 부여하고 관계와 승인 상태를 기록한다.

공식 SAM 3 저장소가 안내하는 현재 추론 환경은 Python 3.12 이상, PyTorch 2.7 이상, CUDA 호환 GPU/CUDA 12.6 이상이며, 모델 체크포인트를 받으려면 Hugging Face 접근 승인이 필요하다. 실제 실행 전에 개발 머신의 GPU·드라이버·Python 환경과 체크포인트 접근 권한을 확인한다. [SAM 3 공식 저장소](https://github.com/facebookresearch/sam3)

## 4. TactileNet 적용안: 생성 접근법과 비교 기준

TactileNet 연구는 Stable Diffusion 1.5 기반의 클래스별 LoRA·DreamBooth 어댑터를 사용해 객체 중심의 2D 촉각 그래픽을 생성한다. 데이터는 66개 객체 범주의 전문가 제작 촉각 그래픽과 프롬프트를 중심으로 구성됐다. 연구는 촉각 전문가 검토를 포함했지만, 생성 결과 중 일부는 큰 수정이 필요했다. 따라서 TactileNet은 **촉각 선화 스타일, 프롬프트 구성, 사람 검수·평가 방식**을 참고하는 데 유용하다. [TactileNet 논문](https://arxiv.org/abs/2504.04722), [공식 GitHub 저장소](https://github.com/Adnan-Khan7/TactileNet)

우리 과제에는 다음과 같은 차이가 있다.

- TactileNet은 객체 중심의 엠보싱용 2D 그래픽을 목표로 한다. TacGen은 원작의 작품 구도, 영역 ID, 촉각 디스플레이 위치 기반 상호작용을 함께 보존해야 한다.
- 공개된 범주형 데이터가 반 고흐의 회화 구도와 같은 복합 작품 장면을 직접 학습했다고 볼 수 없다.
- 확산 모델의 이미지 변환은 중요한 객체의 형태·위치·구성 요소를 바꿀 가능성이 있다. 특히 원본 충실도가 중요한 이번 과제에서는 생성 이미지의 시각적 유사도만으로 품질을 판정할 수 없다.
- 공식 저장소는 학습 데이터와 어댑터 다운로드 및 Stable Diffusion WebUI 기반 이용 절차를 안내하지만, 저장소에 완결된 학습 코드가 제공되는지와 재현 가능성을 실제 도입 전에 별도로 확인해야 한다.
- 데이터셋 카드의 라이선스 표기와 기반 Stable Diffusion 모델, 개별 이미지의 권리 조건은 서로 별도로 확인해야 한다. [TactileNet 데이터셋 카드](https://huggingface.co/datasets/MaiAShaaban/TactileNet)

따라서 TactileNet을 즉시 fine-tuning해 주 생성 모델로 쓰기보다는 아래 순서로 적용한다.

1. 논문·공개 어댑터로 재현 가능한 추론 실험이 되는지 확인한다.
2. SAM 3 분할 결과와 원작을 조건으로 후보 촉각 선화를 만든다.
3. 원본 객체 구성·위치 보존, Region ID 정합, 선화 복잡도와 촉각 판독성을 기준선 렌더러와 비교한다.
4. 작품별 원작-전문가 촉각 그래픽 쌍이 충분히 모이고 권리 확인이 끝난 뒤에만 도메인 적응 학습을 검토한다.

## 5. 권장 통합 파이프라인

```text
작품 이미지 + 공식/큐레이터 자료 + 전문가 촉각 그래픽(확보 시)
  → 입력 정규화, 작품/출처/버전 등록
  → SAM 3 무조건적 후보 생성 또는 선택적 개념 프롬프트 분할
  → 후보 마스크 검수: 누락·중복·과분할·경계 오류 표시
  → 촉각 인지 기준으로 영역 병합/분리/생략 + Object-Region 연결
  → 영역 의미, 공간관계, 출처, 승인 상태 기록
  → 촉각 단순화/60×40 격자 렌더링 + 영역 ID 정합 검사
       ├─ 기준선: 마스크·윤곽 기반 결정론적 변환
       └─ 비교안: TactileNet 계열 선화 생성(선택적)
  → 작품/영역/관계/감각 근거를 유형별 Semantic Chunk로 생성
  → 사용자 입력 접촉점 → 표시 좌표계 → Region ID (+ 인접 후보)
  → Region 우선 검색 + 작품 맥락 보충 검색 + 근거 필터링
  → 답변/음성 안내 + 대상 영역 강조 + 근거/출처 반환
```

### 5.1 입력 등록과 원시 후보 분할

입력 단계에서 `artwork_id`, 작품 제목·작가(미상 허용), 원본 파일 경로·크기·SHA-256, 이미지 출처, 메타데이터 출처와 검수 상태를 등록한다. SAM 3 실행 기록에는 모델 ID·체크포인트/라이브러리 버전, 실행 모드(auto/concept), 프롬프트 설정 파일 및 해시, 임계값, 실행 시각을 남긴다. `auto`는 주제 프롬프트 없이 미분류 후보를 생성하고, `concept`는 사용자가 지정한 개념의 후보를 생성한다.

각 원시 후보는 최종 촉각 영역과 구별되는 `candidate_id`를 가진다. 후보에는 원본 해상도의 이진 마스크(재현 원본), 정규화 폴리곤(미리보기/상호운용), bbox, confidence(없으면 null), 면적, 모델/프롬프트 출처, 후보 상태를 저장한다. 원시 후보는 불변 실행 산출물로 두며, 편집은 별도 최종 Region 객체의 생성/수정으로 표현한다. 따라서 한 후보가 여러 Region으로 나뉘거나 여러 후보가 한 Region으로 합쳐진 이력을 추적할 수 있어야 한다.

### 5.2 Region-Object Segmentation과 촉각 영역 확정

그림 9의 결과 단위는 단순한 SAM 마스크가 아니라, 시각 구조와 촉각 탐색에 의미 있는 **Region**이며, 의미 객체(Object)와 별도 엔터티로 모델링한다. Object는 작품 속 인물/사물/장면 개념이고, Region은 촉각 디스플레이에서 하나의 ID로 탐색·강조할 공간 단위다. 하나의 객체가 여러 부분 Region을 가질 수 있고 하나의 Region이 여러 객체를 묶을 수 있으므로 연결은 다대다로 둔다.

전문가 검수 UI/작업 절차는 오버레이에서 후보를 수락·거부·병합·분할·경계 수정·레이블 수정할 수 있게 한다. 검수자는 각 변경 이유와 사용자 목적을 기록한다. 다음 기준을 Region 구성에 사용한다.

1. **객체 및 의미 영역:** 주요 객체, 배경/바닥/하늘 같은 큰 의미 영역, 작품 이해에 중요한 객체 일부를 선정한다. 붓 터치나 장식 패턴을 기계적으로 모두 별도 영역으로 만들지 않는다.
2. **형태·공간 관계:** 촉각으로 구별 가능한 윤곽과 내부 구성을 남기고, 위/아래·좌/우·안/밖·포함·인접·겹침·연결 등 필요한 관계를 Region/Object 관계 그래프로 기록한다. 모델 추정과 사람 확정을 구분한다.
3. **촉각 식별 가능성:** 60×40 격자로 래스터화한 뒤 최소 면적, 최소 폭/간격, 경계 간격, 고립 점 수를 측정한다. 표현 불가능하거나 서로 붙는 세부는 삭제/병합/확대 후보로 표시하고, 전문가가 선택한다. 자동 기준값은 출력 장치와 사용자 실험으로 보정한다.
4. **ID 안정성:** 최종 확정 전에는 임시 ID를 쓰고, 승인 시 `region_id`를 부여한다. ID는 정렬 순서나 자동 분할의 실행 순서에서 파생하지 않는다. 병합/분할 후 ID 변경은 이전 ID와 새 ID의 대체 관계를 이력으로 남긴다.
5. **검수 게이트:** `candidate → needs_review → approved/rejected`로 흐른다. 미검수 후보는 촉각 최종 출력이나 사실형 답변 근거로 사용하지 않는다. 다만 검수 작업 화면에서는 분할 오류 확인을 위해 표시 가능하다.

최종 Region에는 원본 mask와 정규화 geometry 외에 렌더링에 사용된 `display_mask` 또는 ID 격자 결과를 버전과 함께 기록한다. 원본 geometry, 단순화 geometry, 장치별 출력 geometry를 덮어쓰지 않고 변환 버전을 분리한다.

### 5.3 촉각 그래픽 생성

먼저 색상과 붓 터치에 기대지 않고 Region ID와 윤곽을 나타내는 벡터/격자 기반 결과를 만든다. 이것을 위치 정합과 원본 충실도를 비교할 **기준선**으로 삼는다. TactileNet 계열 생성기는 별도의 선택적 어댑터로 두며, 생성물에서 영역별 ID를 다시 추론해 되살리려 하지 않는다. 원본/영역 마스크의 구조 정보를 보존하는 입력 조건과 정합 검사를 거쳐 생성 결과를 채택한다.

최종 결과에는 동일한 Region ID가 다음 표현에서 일관되게 남아야 한다.

- 표시 장치용 60×40 Region ID 격자와 격자→원본 좌표 변환 정보
- 촉각 그래픽 파일(예: SVG/흑백 래스터) 및 Region별 렌더링 요소
- Material-Sensory Map의 Object/Region 의미, 설명, 관계와 선택적 감각 파라미터

렌더러는 겹치는 영역의 우선순위 규칙, 배경 ID, 축소/확대 방법, 작은 요소 처리와 접근성 출력 형식을 명시한다. 같은 입력/버전에서 같은 결과가 나오는 결정론성을 유지한다. 구조 보존 검사는 출력 격자에서 승인 영역이 사라지지 않았는지, ID가 바뀌지 않았는지, 관계 방향이 역전되지 않았는지 확인한다.

### 5.4 Semantic Chunking, 검색과 촉각 영역 강조

그림 6·17의 Semantic Chunking은 단순 문장 분할이 아니라 **작품 설명과 Region ID를 검색·강조 단위로 연결**하는 작업이다. 자료 원문을 보존한 뒤 출처와 승인 상태를 유지하는 의미 단위로 정규화한다. MVP는 규칙 기반 청크 생성을 사용하고, 벡터 검색/LLM은 승인 자료와 식별자 연결 검증 이후의 선택적 확장으로 둔다.

청크 유형과 분할 규칙은 다음과 같다.

- `artwork_context`: 제목·작가·연도·매체·작품 전체의 공식 설명 및 승인된 큐레이터 해석. 제목/작가 등 단일 필드는 각각 구조화 메타데이터로도 유지한다.
- `object_description`: Object ID별 이름·시각적 근거·작품 내 의미. 각 문장에 대응하는 source/evidence를 유지한다.
- `region_description`: Region ID에 대응하는 촉각 탐색 가능한 형태, 위치, 구성, 질감의 시각적 묘사. 방향 위치는 원본 좌표/인접 Region으로부터 생성한 값인지 큐레이터 기술인지 구분한다.
- `region_relation`: source/target Object·Region ID, 관계 유형과 근거. QA 검색 시 설명 문장에 숨기지 않고 그래프/구조화 필드로 질의할 수 있게 한다.
- `sensory_profile`: 전문가가 제공하고 검수한 상대적 지각 특성 및 장치 출력 힌트. 관찰 사실, 전문가 평가, 장치 명령을 별도 필드로 둔다.
- `source_excerpt`: 검색 가능한 공식 자료의 원문 단위. `source_id`, URL/문서 식별자, 페이지·문단 등 locator와 인용 범위를 보존한다.

청크는 문맥을 유지하는 의미 단위로 나누되 한 청크에서 여러 Region을 언급하면 `region_ids`를 모두 연결한다. 부정확한 자동 공간 관계나 모델 캡션을 승인된 해석처럼 승격하지 않는다. 청크는 `chunk_id`, `kind`, `text`, artwork/object/region IDs, source/evidence locator, approval, language, 생성 방법/버전을 가진다.

검색 라우팅은 (1) 터치 좌표를 장치 좌표계로 보정, (2) 격자 셀에서 Region ID 확인, (3) 경계/배경인 경우 인접 후보를 반환하거나 재탐색 안내, (4) 해당 Region에 직접 연결된 승인 청크 검색, (5) 필요한 경우에만 작품 전체 맥락 청크 추가, (6) 질문 관련성과 출처 권위로 정렬, (7) 근거 부족 시 모른다고 답변 순서로 처리한다. 답변 객체는 `region_id`, 답변 텍스트, `evidence_chunk_ids`, 출처 locator, `highlight_region_id`, polygon/mask 또는 표시 좌표를 반환한다. 멀티모달 모델을 연결하더라도 검색된 승인 근거 바깥의 사실을 추가하지 않도록 한다.

## 6. 데이터 구조: Artwork, Object, Candidate, Region, Semantic Chunk

기존 `examples/artwork.json`은 `artwork`, `display`, `regions`의 최소 실행 예시다. 현재 파이프라인은 Polygon을 래스터화하고 `sources`와 Region의 설명/관계/감각 프로파일을 청크화하지만, Object/Candidate/근거 locator/ID-grid 저장, 변경 이력, 격자 변환 검증은 아직 구현하지 않았다. 따라서 아래 구조를 목표 데이터 계약으로 삼되 단계적으로 확장한다. 대용량 이진 마스크는 JSON에 base64로 넣지 않고 파일 경로와 체크섬으로 참조한다.

```json
{
  "schema_version": "0.3",
  "artwork": {
    "artwork_id": "artwork-001", "title": "작품명", "artist": "작가명",
    "image": {"path": "data/artwork.jpeg", "width": 1200, "height": 900, "sha256": "...", "coordinate_origin": "top_left"},
    "sources": [{"source_id": "museum-001", "kind": "official_description", "uri": "...", "locator": "작품 설명", "text": "...", "approval": "approved"}]
  },
  "segmentation": {
    "segmentation_id": "seg-run-001", "model": "facebook/sam3", "model_version": "...",
    "mode": "auto", "prompt_config": null, "parameters": {"score_threshold": 0.25},
    "created_at": "2026-10-07T00:00:00Z", "status": "needs_curator_review"
  },
  "objects": [{"object_id": "O01", "label": "꽃병", "category": "artifact", "description": "", "approval": "pending", "source_ids": []}],
  "candidates": [{
    "candidate_id": "C001", "segmentation_id": "seg-run-001", "model_label": null,
    "source_prompt": "automatic-mask-generation", "confidence": null,
    "mask_path": "segmentation/artwork/C001.png", "mask_sha256": "...",
    "polygon": [[0.4, 0.2], [0.6, 0.2], [0.6, 0.4]], "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.4],
    "status": "needs_review"
  }],
  "regions": [
    {
      "region_id": "R01", "label": "꽃병", "object_ids": ["O01"], "source_candidate_ids": ["C001"],
      "geometry": {"source_mask_path": "segmentation/artwork/R01.png", "polygon_normalized": [[0.4, 0.2], [0.6, 0.2], [0.6, 0.4]], "display_mask_path": "render/v1/R01.png"},
      "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.4], "description": "화면 중앙 아래쪽의 둥근 꽃병입니다.",
      "tactile_priority": 5, "approval": "approved", "source_ids": ["museum-001"],
      "relations": [{"relation_id": "rel-001", "type": "below", "target_region_id": "R02", "confidence": 1.0, "source": "curator"}],
      "sensory_profile": null
    }
  ],
  "display": {"width": 60, "height": 40, "coordinate_origin": "top_left", "background_id": "0", "overlap_policy": "priority_then_region_id"},
  "render": {"render_id": "render-v1", "region_id_grid_path": "render/v1/region_id_grid.json", "tactile_graphic_path": "render/v1/tactile.svg", "source_geometry_version": "approved-regions-v1"},
  "semantic_chunks": [{
    "chunk_id": "artwork-001:R01:description:v1", "kind": "region_description", "text": "화면 중앙 아래쪽의 둥근 꽃병입니다.",
    "artwork_id": "artwork-001", "object_ids": ["O01"], "region_ids": ["R01"],
    "evidence": [{"source_id": "museum-001", "locator": "문단 2"}], "approval": "approved", "generated_by": "curator", "version": 1
  }]
}
```

좌표 규칙은 모든 geometry에 명시한다. 원본 이미지와 Region 폴리곤은 좌상단 원점의 0~1 정규화 x/y를 사용하고, bbox는 `[x_min,y_min,x_max,y_max]`다. 마스크와 ID 격자는 파일 경로·크기·체크섬·생성 버전을 기록한다. ID-grid 값은 Region ID이며 background는 `0`; 겹침은 정해진 priority와 안정적인 ID 규칙으로 해결한다. 한 작품 JSON을 단일 진실 원천으로 유지하고, 장치별 격자/그래픽은 source geometry version을 가리키는 파생 산출물로 둔다.

검증기는 다음을 거부하거나 경고해야 한다: 중복 Object/Region/Candidate ID, 누락된 참조, 출처 없는 승인 설명, 정규화 범위 밖 geometry, 마스크와 polygon 크기/좌표계 불일치, 연결 대상이 없는 관계, 사용되지 않는 Region ID, 격자에서 사라진 승인 Region, 후보 상태를 승인 상태로 오인한 연결. JSON Schema를 데이터 계약으로 두고 파이프라인 검증기와 함께 CI 테스트한다.

## 7. 단계별 구현 계획

### 단계 A — 현재 프로토타입 기반 다지기

- 현재 JSON/CLI의 실제 계약과 목표 v0.3 계약을 분리해 문서화하고 JSON Schema 초안을 추가한다.
- Artwork, Source, SegmentationRun, Candidate, Object, Region, Relation, SemanticChunk의 ID 및 참조 규칙을 확정한다.
- 정규화 좌표계, bbox 순서, 마스크 파일/체크섬, 버전/변경 이력과 상태 전이를 정한다.
- `sunflower.jpeg`와 두 번째 서로 다른 유형의 작품으로 입력/좌표 계약을 검증한다.
- 완료 조건: 스키마로 유효/무효 fixture를 검증하고, 기존 `examples/artwork.json`을 호환 어댑터 또는 명시적 마이그레이션으로 읽는다.

### 단계 B — SAM 3 분할 실험

- 작품 일반 `auto` 후보와 작품별 `concept` 후보를 같은 입력에서 비교한다. 실행 재현성을 위해 체크포인트와 설정/임계값/환경 버전을 저장한다.
- Candidate 저장과 오버레이를 구현하고, 후보 ID/실행 ID와 최종 Region ID를 분리한다.
- 검수 동작(수락, 거부, 병합, 분할, 경계 수정, 레이블/객체 연결, 관계 추가) 및 변경 이유/행위자를 기록하는 annotation 편집 절차를 구현한다.
- 완료 조건: 검수 결과를 내보냈다가 다시 불러와도 원시 마스크와 수정된 Region, 출처/ID가 보존되고, 다른 작품에 이미지 경로와 메타데이터만 바꿔 같은 파이프라인을 실행할 수 있다.

### 단계 C — 촉각 변환 기준선

- 승인 Region을 대상으로 ID 격자, 접근 가능한 SVG/흑백 raster와 변환 manifest를 생성한다.
- 장치 파라미터(그리드 크기, 최소 점유 셀, 최소 선/간격, 병합·생략 제안, 겹침 정책)를 명시하고 자동 확정 대신 검수 경고를 낸다.
- 완료 조건: 승인된 모든 필수 Region ID가 격자/그래픽/Map에서 추적되고, 소실·ID 혼선·잘못된 정규화 좌표가 없으며, 실제 장치 또는 검증 가능한 촉각 샘플에서 확인한다.

### 단계 D — TactileNet 비교 실험

- 공개 가중치와 재현 절차를 검토하고, 기반 모델·데이터·어댑터별 이용 조건을 확인한다.
- 동일한 입력/영역 조건으로 후보 선화를 생성해 기준선과 나란히 비교한다.
- 원본 객체 구성, 주요 위치, 영역 의미, 복잡도, 선화 판독성을 각각 채점한다.
- 실제 작품-전문가 촉각 그래픽 쌍과 충분한 데이터가 확보되기 전에는 자체 학습을 보류한다.

### 단계 E — 작품 맥락 질의응답 연결

- 작품 공식 자료, 큐레이터 기술, 전문가 감각 정보를 서로 다른 Source kind로 등록하고 locator/권한/승인 상태를 유지한다.
- `artwork_context`, `object_description`, `region_description`, `region_relation`, `sensory_profile` 청크를 만들고 소스/객체/영역 ID와 다대다 연결한다.
- 터치점(또는 영역 선택) → 좌표 보정 → Region lookup → Region 우선 검색 → 작품 맥락 보충 → 근거 포함 답변/영역 강조 API를 구현한다.
- 완료 조건: 검색된 근거가 실제 답변에 제시되고, 미승인 자료를 쓰지 않으며, 매칭 실패/자료 없음 시 명시적으로 안내한다. 강조 geometry와 응답 Region ID가 동일해야 한다.

## 8. 검증 계획과 통과 기준

### 기술 검증

- 데이터 계약: 중복 ID, 유효하지 않은 상태 전이, 참조가 끊긴 Object/Region/Source/Chunk, 범위 밖 좌표, 없는 관계 대상, 출처 없는 승인 콘텐츠가 오류로 잡힌다. 단위/통합 테스트에서 마스크-폴리곤-격자 좌표 변환 round trip을 확인한다.
- 분할 품질: 전문가 정답과 Candidate 수준의 mask IoU/경계 F-score를 평가하고, 최종 Region 수준에서는 객체/영역 F1, 관계 정확도와 촉각 중요 객체 recall을 측정한다. SAM 결과, 검수 후 결과를 별도로 기록한다.
- 촉각 적합성: 60×40 실제 출력에서 영역 소실률, 인접 영역 합침률, 최소 폭/간격 위반, 전문가의 수정량을 기록한다. 출력 장치가 정해지기 전 수치는 탐색용으로만 사용한다.
- 렌더링/정합: 모든 승인 Region이 존재하며 Region ID가 SVG, Map, ID 격자, semantic chunks 및 강조 응답에서 일치한다. 변경 이력으로 mask와 최종 geometry 관계를 재현한다.
- 접촉 매칭: 내부/경계/배경 표본을 나눠 Region ID 정확도, 경계 오차 허용률, 재탐색 유도 성공률을 기록한다. 제안서의 90% 목표는 평가 데이터/표본 수/성공 정의를 정한 뒤 판단한다.
- Semantic retrieval/QA: Region hit@k, 근거 출처 정확성, 지원되지 않는 주장 비율, 미승인 근거 사용률(목표 0), evidence 없는 답변의 올바른 거절률을 측정한다.
- 일반화: 최소 2개 이상의 주제/구도 유형을 포함한 held-out 작품에서 같은 auto 설정을 적용한다. 한 작품/한 모델 실행은 기술 작동 확인일 뿐 일반화의 증거로 쓰지 않는다.

### 촉각 품질 평가

기준선 렌더러와 TactileNet 후보를 같은 작품·영역에서 비교한다. 객체/윤곽 보존, 공간관계 보존, 과밀도, 구별 가능성, 촉각 탐색 성공률을 기록한다. 전문가 검토 후 시각장애인·저시력 사용자 평가를 수행하고, 생성 이미지 유사도나 자동 점수만으로 사용성을 대체하지 않는다.

## 9. 성공 기준과 위험 관리

프로토타입 성공은 SAM 3가 마스크를 냈다는 사실만으로 판정하지 않는다. 검수 가능한 영역 ID가 생성되고, 촉각 그래픽과 영역 Map 사이의 정합이 유지되며, 사용자가 실제로 구별·탐색할 수 있고, 영역 기반 답변이 승인된 작품 자료에 근거해야 한다.

주요 위험과 대응은 다음과 같다.

- **회화의 붓 터치·겹침으로 인한 분할 오류:** 프롬프트 분리, 마스크 오버레이 검토, 전문가 병합·수정으로 대응한다.
- **60×40 표시에서의 세부 소실:** 작은 꽃잎·질감을 모두 보존하지 않고 의미 단위로 병합·생략한다.
- **생성 모델의 구성 변경/환각:** 영역 마스크 기준선과 비교하고 구조 정합 검사를 통과하지 못한 생성물은 제외한다.
- **학습 데이터 부족:** 현재 한 장의 이미지는 학습 자료가 아니다. 다양한 원작-전문가 촉각 그래픽 쌍과 사용권 확보 후 학습을 판단한다.
- **모델 환경과 접근 제약:** 체크포인트 접근 승인, GPU/드라이버, 기반 모델·데이터 라이선스를 선행 확인한다.

## 10. 현재 코드 상태

현재 TacGen 코드에는 폴리곤 기반 최소 입력 검증, Region ID 격자 렌더링, 작품 출처와 Region description/relation/sensory profile의 규칙 기반 Semantic Chunk 생성, 좌표 조회, 승인 청크를 사용하는 발췌형 답변/영역 하이라이트가 있다. SAM 3 어댑터(`scripts/segment_sam3.py`)는 기본 자동 후보 모드와 선택적 개념 프롬프트 모드를 지원하고, JSON·마스크·오버레이를 내보낸다. `data/artwork.json`은 해바라기 26개 후보가 `pending` 상태인 실제 실행 산출물이며, 의미 객체 엔터티, 검수 편집 이력, 안정적 Region ID 확정, Object-Region 다대다 연결, 관계 어노테이션, Semantic Chunk의 원문 locator/근거, ID 격자 저장, 실제 촉각 렌더러는 아직 구현되지 않았다. 현재 `tacgen.pipeline.semantic_chunks`는 구조화된 단순 청크 생성이며, 일반 검색 인덱스/벡터 RAG나 생성형 QA가 아니다. 다음 구현은 목표 v0.3 데이터 계약과 검증기를 먼저 추가하고, Candidate→검수 Region 편집 흐름과 촉각/QA 정합 평가를 단계별로 구현하는 것이다. TactileNet 생성기는 아직 연결되지 않았다.
