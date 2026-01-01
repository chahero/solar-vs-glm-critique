# LayerNorm Similarity: Additional Experiments

## 📋 실험 결과 요약

이 레포지토리는 solar-vs-glm 실험에 대한 **추가 실험**을 제공합니다. 데이터 해석은 독자 여러분께 맡기겠습니다.

| 실험 | 결과 |
|------|------|
| Layer 0 baseline | ~0.38 |
| 인접 레이어 간 유사도 | ~0.99 |
| Solar vs GLM | 0.981 |
| MoE 모델 간 평균 | 0.964 |
| MoE vs non-MoE | 0.972 |
| non-MoE vs non-MoE | 0.974 |

**관측된 패턴:**
- Layer 0 기준 유사도: ~0.38
- 인접 레이어(10↔20, 20↔30) 유사도: ~0.99
- hidden_size=4096인 모델들의 cross-model 유사도: ~0.97

---

## 실험 목적

이 실험은 solar-vs-glm 연구에서 사용된 **"Layer 0 baseline" 방법론**에 대한 추가 검증을 제공합니다.

### 배경

solar-vs-glm 레포지토리는 다음과 같이 주장합니다:

```
Within-model baseline (Layer 0 vs 10,20,30,40): 0.377
Cross-model (Solar vs GLM, same layer):         0.989
차이: 0.612 (182 시그마)
결론: Solar는 GLM에서 파생되었다
```

### 추가 검증 포인트

1. **Layer 0의 특성**
   - 토크나이저 확장 영향을 직접 받음
   - 다른 레이어와 패턴이 다를 수 있음

2. **비교 조건**
   - Within: Layer 0 vs 10,20,30,40 (10~40칸 차이)
   - Cross: Solar[10] vs GLM[10] (0칸 차이, 같은 위치)

3. **추가 통제군**
   - 독립적인 제3의 모델들과의 비교

---

## 실험 방법

### 세 가지 비교

**1. solar-vs-glm 방식 (Layer 0 baseline)**
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

Difference (fair vs cross):     ~0.00
Difference (Layer 0 vs cross):  ~0.60
```

### 관측 결과

- **Layer 0 baseline (0.38)**: 다른 레이어 대비 낮은 유사도
- **Fair baseline (0.98)**: 인접 레이어 간 유사도
- **Cross-model (0.98)**: Fair baseline과 유사한 수준

---

## 시각화 결과

### 요약 (4-Panel Overview)
![Summary](results/summary_comparison.png)

### 개별 실험 결과

**실험 1: Layer 0은 Outlier**
![Layer 0 Outlier](results/exp1_layer0_outlier.png)

**실험 2: 인접 레이어는 높은 유사도**
![Fair Baseline](results/exp2_fair_baseline.png)

**실험 3: MoE 모델 간 유사도 (Layer 10)**
![MoE Heatmap](results/exp3_moe_heatmap.png)

**실험 4: 레이어별 유사도 일관성**
![Multi-layer](results/exp4_multi_layer.png)

**실험 5: 아키텍처별 비교 (hidden_size=4096)**
![Architecture Comparison](results/exp5_architecture_comparison.png)

**전체 9개 모델 유사도 매트릭스**
![Full Heatmap](results/exp5_full_heatmap.png)

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
solar-vs-glm-critique/
├── .gitignore                   # Git ignore 설정
├── README.md                    # 이 파일
├── compare_baselines.py         # 메인 실험 스크립트 (5개 실험 포함)
├── cache/                       # 다운로드한 LayerNorm 캐시 (실행 후 생성)
│   ├── upstage_Solar-Open-100B_layer0_input_layernorm.npy
│   ├── zai-org_GLM-4.5-Air_layer0_input_layernorm.npy
│   └── ...
└── results/                     # 시각화 결과 (실행 후 생성)
    ├── RESULTS.md               # 실험 결과 상세 리포트
    ├── summary_comparison.png   # 4-Panel 요약 이미지
    ├── exp1_layer0_outlier.png  # 실험 1: Layer 0 Outlier
    ├── exp2_fair_baseline.png   # 실험 2: 인접 레이어 유사도
    ├── exp3_moe_heatmap.png     # 실험 3: MoE 모델 히트맵
    ├── exp4_multi_layer.png     # 실험 4: 레이어별 일관성
    ├── exp5_architecture_comparison.png  # 실험 5: 아키텍처 비교
    └── exp5_full_heatmap.png    # 9개 모델 전체 히트맵
```

---

## 인용

이 실험은 다음 레포지토리의 주장을 검증합니다:

- **solar-vs-glm**: https://github.com/sionic-ai/solar-vs-glm
- **solar-vs-glm-vs-phi**: https://github.com/hyunwoongko/solar-vs-glm-vs-phi

---

## 라이선스

이 코드는 교육 및 연구 목적으로 자유롭게 사용 가능합니다.

---

위 실험 결과의 해석은 독자 여러분께 맡기겠습니다.
