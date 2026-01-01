# Baseline Critique: Why Layer 0 is an Inappropriate Baseline

## 📋 진행 상황

- ✅ **Experiment 1**: Layer 0 baseline 문제 증명 (완료)
  - `compare_baselines.py` 작성 및 실행 완료
  - 결과: `results/baseline_comparison.png`
  - 결론: Layer 0은 outlier, Fair baseline과 Cross-model은 거의 동일 (차이 1.4%p)

- ⏳ **Experiment 2**: 5개 MoE 모델 비교 (스크립트 작성 완료, 실행 대기)
  - `compare_multiple_models.py` 작성 완료
  - 모델: Solar, GLM, Phi, Mixtral-8x7B, Mixtral-8x22B
  - 목적: "MoE는 원래 LayerNorm이 다 비슷함" 증명

- 🔜 **다음 단계 (고려 중)**
  - Option A: Dense 모델 추가 (Llama, Qwen 등)
  - Option B: 전체 레이어 히트맵
  - Option C: Attention/MLP 가중치 검증

---

## 실험 목적

이 실험은 **solar-vs-glm의 "Layer 0 baseline" 방법론이 부적절함**을 증명합니다.

### 배경

solar-vs-glm 레포지토리는 다음과 같이 주장합니다:

```
Within-model baseline (Layer 0 vs 10,20,30,40): 0.377
Cross-model (Solar vs GLM, same layer):         0.989
차이: 0.612 (182 시그마)
결론: Solar는 GLM에서 파생되었다
```

### 문제점

1. **Layer 0은 특이값 (outlier)**
   - 토크나이저 확장 영향을 직접 받음
   - 다른 레이어와 패턴이 다름

2. **비교 조건이 불공정**
   - Within: Layer 0 vs 10,20,30,40 (10~40칸 차이)
   - Cross: Solar[10] vs GLM[10] (0칸 차이, 같은 위치)

3. **통제군 부재**
   - 독립적인 세 번째 모델(Phi) 없음

---

## 실험 방법

### 세 가지 비교

**1. solar-vs-glm 방식 (부적절한 baseline)**
```
Solar: Layer 0 vs Layer 10, 20, 30, 40
GLM:   Layer 0 vs Layer 10, 20, 30, 40
```

**2. 공정한 baseline (같은 거리)**
```
Solar: Layer 10 vs Layer 20 (10칸 차이)
       Layer 20 vs Layer 30 (10칸 차이)
GLM:   동일
```

**3. Cross-model (같은 위치)**
```
Solar[10] vs GLM[10]
Solar[10] vs Phi[10]
GLM[10] vs Phi[10]
```

---

## GPU 요구사항

**불필요!** 이 실험은 CPU만으로 실행 가능합니다.

- HTTP Range request로 LayerNorm만 다운로드 (~수 MB)
- 전체 모델을 다운로드하지 않음 (~100GB)
- RAM: 4GB 이상 권장
- 실행 시간: 5-10분

---

## 실행 방법

### 1. 환경 설정

```bash
# Python 3.8 이상 필요

# 필요한 패키지 설치
pip install numpy requests matplotlib seaborn
```

### 2. 실험 실행

**Experiment 1: Layer 0 baseline 검증**
```bash
python compare_baselines.py
# 결과: results/baseline_comparison.png
```

**Experiment 2: 5개 MoE 모델 비교**
```bash
python compare_multiple_models.py
# 결과: results/multi_model_comparison.png
```

### 3. 결과 확인

```
cache/                           # 다운로드한 LayerNorm 캐시 (~수 MB)
results/baseline_comparison.png  # 시각화 결과
```

**캐싱**: 다운로드한 LayerNorm은 `cache/` 폴더에 저장되어 다음 실행 시 재사용됩니다.

---

## 예상 결과

```
1. solar-vs-glm baseline (Layer 0):  ~0.38
2. Fair baseline (adjacent layers):  ~0.98
3. Cross-model (Solar vs GLM):       ~0.98

Difference (fair vs cross):     ~0.00 (거의 없음!)
Difference (Layer 0 vs cross):  ~0.60 (인위적으로 큼)
```

### 해석

- **Layer 0 baseline (0.38)**: 토크나이저 확장 때문에 인위적으로 낮음
- **Fair baseline (0.98)**: 정상적인 within-model 유사도
- **Cross-model (0.98)**: Fair baseline과 거의 차이 없음

**결론:** Layer 0를 baseline으로 쓰면 차이가 과장됩니다!

---

## 시각화 예시

![Baseline Comparison](results/baseline_comparison.png)

그래프 설명:
- **회색 막대**: Layer 0 baseline (부적절)
- **파란색 막대**: 공정한 baseline (인접 레이어)
- **빨간색 막대**: Cross-model 비교

파란색과 빨간색이 비슷하면 → LayerNorm은 원래 다 비슷함
회색이 낮으면 → Layer 0이 특이값임

---

## 기술적 세부사항

### HTTP Range Request

전체 모델 파일(~100GB)을 다운로드하지 않고, LayerNorm weight만 선택적으로 다운로드:

```python
# LayerNorm 크기: 4096 × 2 bytes (FP16) = 8KB
# 전체 모델 대비 0.00001% 미만!

def get_layernorm_weight(repo_id, layer_idx, ln_type):
    # 1. safetensors 헤더만 다운로드 (수 KB)
    # 2. LayerNorm 위치(offset) 파악
    # 3. 해당 부분만 HTTP Range request
    # 4. numpy array로 변환
```

### Cosine Similarity 계산

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## FAQ

### Q1: 왜 Layer 0이 특이한가?

**A:** Layer 0은 임베딩 레이어 바로 다음입니다:
```
Embedding (vocab_size 변경: 151K → 196K)
    ↓ 토큰 분포 변화
Layer 0 LayerNorm ← 직접 영향!
    ↓
Layer 1 LayerNorm ← 영향 감소
    ↓
Layer 5+ LayerNorm ← 영향 최소화
```

### Q2: "같은 거리" 비교가 왜 공정한가?

**A:**
- solar-vs-glm: Layer 0 vs 10 (10칸 차이)와 Solar[10] vs GLM[10] (0칸) 비교 → 불공정
- 공정한 방법: 둘 다 10칸 차이로 비교 (Layer 10 vs 20)

### Q3: Phi 모델이 왜 중요한가?

**A:** 통제군(control group)입니다:
- Solar-GLM만 비교하면 → "둘이 비슷하네" (파생 가능성)
- Solar-Phi, GLM-Phi도 비교하면 → "셋 다 비슷하네" (원래 다 비슷함)

### Q4: 정말 GPU 없이 되나?

**A:** 네! HTTP Range request 덕분에:
- 다운로드: ~10MB (LayerNorm만)
- 메모리: ~100MB
- CPU 계산: 수십 초

---

## 파일 구조

```
baseline-critique/
├── .gitignore                   # Git ignore 설정
├── README.md                    # 이 파일
├── compare_baselines.py         # 메인 실험 스크립트
├── cache/                       # 다운로드한 LayerNorm 캐시 (실행 후 생성)
│   ├── upstage_Solar-Open-100B_layer0_input_layernorm.npy
│   ├── zai-org_GLM-4.5-Air_layer0_input_layernorm.npy
│   └── ...
└── results/
    └── baseline_comparison.png  # 시각화 결과 (실행 후 생성)
```

---

## 인용

이 실험은 다음 레포지토리의 주장을 검증합니다:

- **solar-vs-glm**: https://github.com/sionic-ai/solar-vs-glm
- **solar-vs-glm-vs-phi**: (현재 폴더 상위 디렉토리)

---

## 라이선스

이 코드는 교육 및 연구 목적으로 자유롭게 사용 가능합니다.

---

**결론:** Layer 0 baseline은 cherry-picking입니다. 공정한 비교를 하면 Solar와 GLM의 LayerNorm 유사도는 일반적인 범위 내에 있습니다.
