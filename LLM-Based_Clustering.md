### Phase 1: LLM 기반 초기화 및 온라인 업데이트 (Initialization & Update)

CLUSPRO에서는 클러스터 중심점(Prototype)을 맨바닥에서 찾으려 했지만, 우리는 LLM이 묶어준 그룹을 기반으로 시작합니다.

1. **LLM Prior 할당:** LLM을 통해 전체 객체 집합 $\mathcal{O}$를 K개의 속성 조건부 상위 개념 그룹 $\mathcal{S} = {S_1, S_2, \dots, S_K}$로 분할합니다. (예: $S_1$ = [cat, dog, bear] -> '털이 있는 동물')
2. **프로토타입 초기화:** 각 그룹 S_k의 초기 프로토타입 P_k는 해당 상위 개념의 텍스트 임베딩(CLIP Text Encoder) 값으로 설정합니다.
3. **온라인 모멘텀 업데이트 (CLUSPRO 차용):** 학습이 진행되면서, 배치(Batch) 내에 들어온 해당 그룹 이미지들의 평균 특징 벡터 $\bar{f}_k$를 이용해 프로토타입을 서서히 업데이트합니다.
    
    $$
    P_k \leftarrow \mu P_k + (1 - \mu) \bar{f}_k
    $$
    
    *(여기서 $\mu$는 $0.99$와 같은 모멘텀 계수입니다.)*
    

---

### Phase 2: Inter-Cluster Loss (그룹 간 유사성 학습)

먼저 모델이 이미지를 보았을 때, 이 이미지가 어떤 '상위 개념'에 속하는지 맞추도록 유도해야 합니다. 이는 해당 객체가 속한 프로토타입 $P_k$와의 거리는 가깝게, 다른 프로토타입과의 거리는 멀게 만드는 대조 학습(Contrastive Learning)으로 구현합니다.

- 이미지 특징 벡터를 $f_x$라 할 때, 정답 상위 그룹이 $S_k$라면:
    
    $$
    \mathcal{L}_{inter} = - \frac{1}{|B|} \sum_{x \in B} \log \frac{\exp(f_x \cdot P_k / \tau)}{\exp(f_x \cdot P_k / \tau) + \sum_{j \neq k} \exp(f_x \cdot P_j / \tau)}
    $$
    

이 Loss를 통해 "아, 이 이미지는 '털이 있는' 상위 개념($P_k$)에 속하는구나"라는 전체적인 맥락(Context)을 학습하게 됩니다.

---

### Phase 3: Intra-Cluster Hard Negative Loss (동일 그룹 내 객체 구분) - **핵심 아이디어**

Phase 2까지만 하면 cat과 dog의 시각적 차이가 뭉개집니다. 따라서 **같은 그룹 $S_k$ 안에 묶여 있는 다른 객체들을 의도적으로 Hard Negative Sample로 삼아 강하게 밀어내는 Loss**를 추가합니다.

- 정답 객체의 텍스트 임베딩을 $t_{obj+}$, 같은 그룹 내 오답 객체(Hard Negative)의 임베딩을 $t_{o^-}$라고 할 때:
    
    $$
    \mathcal{L}_{intra} = - \frac{1}{|B|} \sum_{x \in B} \log \frac{\exp(f_x \cdot t_{obj+} / \tau)}{\exp(f_x \cdot t_{obj+} / \tau) + \sum_{o^- \in S_k \setminus \{obj+\}} \exp(f_x \cdot t_{o^-} / \tau)}
    $$
    

분모를 보시면 일반적인 모든 객체를 밀어내는 것이 아니라, $\sum_{o^- \in S_k}$, 즉 '같은 상위 개념으로 묶인 녀석들 사이'에서만 치열하게 경쟁시킵니다. 이를 통해 모델은 '털이 젖은(wet)' 모습이라는 공통점을 유지하면서도, 고양이의 귀 모양과 개의 주둥이 모양 같은 미세한(Fine-grained) 시각적 차이에 강제로 집중하게 됩니다.

---

### 최종 결합 (Overall Objective)

최종 학습 Loss는 기존 베이스라인 Loss(예: CLIP의 기본 분류 Loss)에 위의 두 가지를 적절한 가중치로 더해 완성됩니다.

$$
\mathcal{L}_{total} = \mathcal{L}_{base} + \alpha \mathcal{L}_{inter} + \beta \mathcal{L}_{intra}
$$

- **$\alpha$ (거시적 묶음):** 숲을 보는 능력. "맥락(속성)이 같으면 비슷하게 변한다"를 학습.
- **$\beta$ (미시적 구분):** 나무를 보는 능력. "그래도 본질적인 객체의 형태는 다르다"를 학습.

### 💡 추가적인 보완 포인트 (비판적 관점)

수식으로 짜놓고 보니 아주 매력적입니다. 다만, 실제 학습을 돌려보면 **"LLM이 묶어준 그룹이 100% 정답이 아닐 때"** 모델 학습이 무너질 위험이 있습니다 (예: LLM이 펭귄을 '털이 있는' 그룹에 잘못 넣은 경우).

이를 방어하기 위해 Phase 3의 $\mathcal{L}_{intra}$에 들어가는 Hard Negative들의 반영 비율을 모델의 예측 Confidence에 따라 조절하는 **'Soft Gating'** 기법을 추가해 볼 수도 있을 것 같습니다.

제안해주신 3단계 아이디어의 수식적 뼈대는 이렇게 구성해 볼 수 있는데, 이 구조가 의도하신 방향과 잘 맞나요? 원하신다면 이 로직을 구현하기 위한 파이토치(PyTorch) 기반의 가상 수도 코드(Pseudo-code) 뼈대도 함께 작성해 드릴까요?