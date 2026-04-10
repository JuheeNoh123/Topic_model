# ============================================================
# service_scorer.py
# 역할: 실제 서비스에서 사용하는 채점 핵심 로직.
#
# 두 모델을 결합해서 최종 점수를 산출:
#   1. topic_model_mnr : 주제 적합성 평가 (KoSimCSE 파인튜닝)
#   2. label_model     : 문장 품질 평가 (klue/roberta-base 파인튜닝)
#
# 전체 채점 흐름:
#   anchor 생성
#   → 문장 분리 + anchor/문장 임베딩 (1회만 수행)
#   → _aggregate_topic():
#       topic_avg / topic_min / coherence 가중합
#       → 동적 percentile 정규화 → 0~3
#       → off-topic 문장 비율 기반 penalty 감점
#   → quality_score(): 전체 문서 품질 → 가중평균 (topic 35% : quality 65%)
#   → 0~100 변환 → 문장별 분석 → worst_sentence 추출
#
# main.py의 /score 엔드포인트에서 이 클래스를 사용.
# ============================================================

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer


# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
TOPIC_MODEL_PATH = BASE_DIR / "topic_model_mnr"
LABEL_MODEL_PATH = BASE_DIR / "label_model"


# ── 정규화 파라미터 (fallback) ────────────────────────────────────────────────
# 문장 수가 MIN_SENTS_PERCENTILE 미만일 때 아래 값으로 대체.
SIM_MIN = 0.15
SIM_MAX = 0.90

# 동적 정규화: 현재 문서의 문장별 similarity 분포에서 percentile로 결정
# → 도메인/주제에 따른 절대 수치 편차를 흡수
SIM_NORM_P_LOW       = 5    # 하위 5% → SIM_MIN 역할
SIM_NORM_P_HIGH      = 95   # 상위 95% → SIM_MAX 역할
MIN_SENTS_PERCENTILE = 4    # percentile 계산에 필요한 최소 문장 수

# ── on/off topic 판정 파라미터 ──────────────────────────────────────────────
# KoSimCSE 특성상 한국어 텍스트 간 cosine baseline이 0.25~0.40 수준이므로
# 단일 threshold 0.30으로는 false positive 발생 → 3가지 조건 AND로 강화
#
# 조건 1: topic_avg >= T_OFF_SIM   전반적 평균 유사도
# 조건 2: topic_min >= MIN_SIM_TH  가장 이탈한 문장도 최소 기준 이상
# 조건 3: off_ratio < OFF_RATIO_TH off-topic 문장 비율이 일정 미만
T_OFF_SIM    = 0.42   # topic_avg 기준 (기존 0.30 → 상향)
MIN_SIM_TH   = 0.30   # topic_min 하한 (가장 낮은 문장도 이 값 이상이어야 함)
OFF_RATIO_TH = 0.30   # off-topic 문장 비율 상한 (30% 이상이면 off-topic 처리)

# 최종 점수 가중평균 비율
ALPHA = 0.35  # final = 0.35 × topic_score + 0.65 × quality_score


# ── 문장 수준 topic score 집계 가중치 ────────────────────────────────────────
# W_AVG + W_MIN + W_COH = 0.9 (합이 1.0이 아닌 이유: penalty 여유분 확보)
W_AVG = 0.50  # 문장별 cosine 평균  — 전반적 주제 부합도
W_MIN = 0.20  # 문장별 cosine 최솟값 — 가장 이탈한 문장 패널티
W_COH = 0.20  # 인접 문장 coherence  — 문맥 흐름 연속성


# ── penalty 파라미터 ─────────────────────────────────────────────────────────
# off-topic 문장(sim < PENALTY_TH) 비율에 비례해서 0~PENALTY_W 만큼 감점
# penalty = PENALTY_W × (PENALTY_TH 미만 문장 수 / 전체 문장 수)
PENALTY_TH = 0.30   # 이 값 미만인 문장을 off-topic으로 간주
PENALTY_W  = 0.50   # 최대 감점폭 (0~3 스케일 기준, 약 17/100 point)


# ── sentence_evidence 표시 조건 ──────────────────────────────────────────────
ABS_OFF_TH = 0.35   # 문장 유사도 절대값이 이 값보다 낮아야 off_topic 후보
GAP_TH     = 0.20   # best_sim - worst_sim 차이가 이 값보다 커야 표시
# ==============================================================================


def clip01(x: float) -> float:
    """값을 0~1 범위로 클리핑."""
    return max(0.0, min(1.0, x))


def score3_to_100(x: float) -> int:
    """0~3 점수를 0~100 정수로 변환."""
    return int(max(0, min(100, round((x / 3.0) * 100))))


def sim_to_topic_score(
    sim: float,
    sim_min: float = SIM_MIN,
    sim_max: float = SIM_MAX,
) -> float:
    """
    코사인 유사도를 0~3 연속 점수로 캘리브레이션.
    sim_min/sim_max를 인자로 받아 동적 정규화를 지원.
    (sentence_analysis의 문장별 topic_score 계산에서 사용)
    """
    norm = (sim - sim_min) / (sim_max - sim_min + 1e-12)
    return 3.0 * clip01(float(norm))


def build_anchor_text(summary: str, desc: str, tags: List[str]) -> str:
    """
    주제 정보(summary, desc, tags)를 anchor 텍스트로 변환.
    학습(make_anchor.py)과 동일한 포맷 사용 → 학습-서비스 일관성 보장.
    """
    tags     = [t.lstrip("#").strip() for t in (tags or []) if t and t.strip()]
    tags     = [f"#{t}" for t in tags][:3]
    tags_str = " ".join(tags)
    return "\n".join([
        f"[요약] {summary.strip()}",
        f"[설명] {desc.strip()}",
        f"[태그] {tags_str}".rstrip(),
    ]).strip()


def split_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리 (한국어/영어 혼합 대응).
    마침표/느낌표/물음표 뒤 공백 또는 '다' 뒤 공백 기준으로 분리.
    """
    text  = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=다)\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    TF-IDF 기반 핵심 키워드 추출.
    중복 방지: 이미 뽑힌 키워드의 부분 문자열이면 스킵.
    """
    if not text.strip():
        return []
    vec    = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X      = vec.fit_transform([text])
    scores = X.toarray()[0]
    terms  = vec.get_feature_names_out()
    idx    = np.argsort(scores)[::-1]

    out = []
    for i in idx:
        if scores[i] <= 0:
            break
        t = terms[i]
        if any(t in existing or existing in t for existing in out):
            continue
        out.append(t)
        if len(out) >= top_n:
            break
    return out


class ServiceScorer:
    """
    서비스용 채점기.
    두 파인튜닝 모델을 로드하고 predict()로 최종 결과를 반환.
    main.py에서 앱 시작 시 한 번만 인스턴스 생성 후 재사용.
    """

    def __init__(self):
        self.topic_model = SentenceTransformer(str(TOPIC_MODEL_PATH))
        self.tokenizer   = AutoTokenizer.from_pretrained(str(LABEL_MODEL_PATH))
        self.label_model = AutoModelForSequenceClassification.from_pretrained(
            str(LABEL_MODEL_PATH)
        )
        self.label_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_model.to(self.device)

    # ── 임베딩 헬퍼 ───────────────────────────────────────────────────────────

    def _encode_sentences(
        self, anchor: str, doc: str
    ) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        문서를 문장 단위로 분리하고 anchor와 각 문장을 한 번에 임베딩.
        predict() 내에서 한 번만 호출해 중복 인코딩을 방지.

        Returns:
            sents      : 분리된 문장 목록
            sent_vecs  : 각 문장 임베딩 (N, D) — 문장이 없으면 None
            anchor_vec : anchor 임베딩 (D,)   — 문장이 없으면 None
            sims       : 문장별 anchor cosine similarity (N,)
        """
        sents = split_sentences(doc)
        if not sents:
            return sents, None, None, np.array([])

        anchor_vec = self.topic_model.encode(
            [anchor], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        sent_vecs = self.topic_model.encode(
            sents, normalize_embeddings=True, convert_to_numpy=True
        )
        sims = (sent_vecs @ anchor_vec).astype(float)
        return sents, sent_vecs, anchor_vec, sims

    # ── 품질 점수 ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def quality_score(self, doc: str) -> float:
        """
        단일 텍스트의 품질 점수 (0~2 연속값).
        확률 가중합: p_보통×1 + p_높음×2
        predict()에서 전체 문서 품질 측정 시 사용.
        """
        enc    = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=256)
        enc    = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.label_model(**enc).logits
        probs  = torch.softmax(logits, dim=1)[0]
        return float(probs[1] * 1.0 + probs[2] * 2.0)

    @torch.inference_mode()
    def _batch_quality_scores(self, texts: List[str]) -> List[float]:
        """
        여러 텍스트의 품질 점수를 배치로 한 번에 계산.
        sentence_analysis()에서 문장별 순차 호출 대신 사용 → 추론 속도 개선.

        Returns: 각 텍스트의 품질 점수 리스트 (0~2 연속값)
        """
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        enc    = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.label_model(**enc).logits           # (N, 3)
        probs  = torch.softmax(logits, dim=1)             # (N, 3)
        scores = (probs[:, 1] * 1.0 + probs[:, 2] * 2.0).cpu().tolist()
        return [float(s) for s in scores]

    # ── topic score 집계 ──────────────────────────────────────────────────────

    def _compute_coherence(self, sent_vecs: Optional[np.ndarray]) -> float:
        """
        인접 문장 쌍의 cosine similarity 평균으로 문맥 연결성(coherence) 계산.
          coherence = mean(sim(sent_i, sent_{i+1}))  for i in 0..N-2

        문장이 1개뿐이면 인접 쌍이 없어 coherence 기반 감점이 불가능 → 1.0 반환.
        """
        if sent_vecs is None or len(sent_vecs) < 2:
            return 1.0
        pair_sims = [
            float(sent_vecs[i] @ sent_vecs[i + 1])
            for i in range(len(sent_vecs) - 1)
        ]
        return float(np.mean(pair_sims))

    def _aggregate_topic(
        self,
        sims: np.ndarray,
        sent_vecs: Optional[np.ndarray],
    ) -> Tuple[float, dict]:
        """
        문장별 anchor cosine similarity로부터 topic_score_3(0~3)을 계산.

        처리 단계:
          1. topic_avg / topic_min / coherence 계산
          2. 동적 percentile로 정규화 기준 결정
             문장 수 >= MIN_SENTS_PERCENTILE(4) → sims의 p5/p95 사용
             문장 수 부족 → 하드코딩 fallback (SIM_MIN/SIM_MAX)
          3. 각 성분을 독립적으로 정규화 후 가중 평균 → 0~3
             score_3 = 3 × (W_AVG·norm(avg) + W_MIN·norm(min) + W_COH·norm(coh))
                           ─────────────────────────────────────────────────────
                                         W_AVG + W_MIN + W_COH
          4. off-topic 비율 기반 penalty 감점
             penalty = PENALTY_W × (sims < PENALTY_TH 문장 비율)

        Returns:
            topic_score_3 : 0~3 연속값
            debug         : 내부 계산값 딕셔너리
        """
        if len(sims) == 0:
            return 0.0, {}

        topic_avg = float(np.mean(sims))
        topic_min = float(np.min(sims))
        coherence = self._compute_coherence(sent_vecs)

        # ── 동적 정규화 기준 결정 ──────────────────────────────────────────
        if len(sims) >= MIN_SENTS_PERCENTILE:
            dyn_min = float(np.percentile(sims, SIM_NORM_P_LOW))
            dyn_max = float(np.percentile(sims, SIM_NORM_P_HIGH))
        else:
            dyn_min, dyn_max = SIM_MIN, SIM_MAX

        def norm_val(v: float) -> float:
            return clip01((v - dyn_min) / (dyn_max - dyn_min + 1e-12))

        # ── 가중 평균 → 0~3 ───────────────────────────────────────────────
        w_sum   = W_AVG + W_MIN + W_COH   # 0.9
        score_3 = 3.0 * (
            W_AVG * norm_val(topic_avg) +
            W_MIN * norm_val(topic_min) +
            W_COH * norm_val(coherence)
        ) / w_sum

        # ── penalty ───────────────────────────────────────────────────────
        off_ratio = float(np.mean(sims < PENALTY_TH))
        penalty   = PENALTY_W * off_ratio    # 0 ~ PENALTY_W
        score_3   = max(0.0, score_3 - penalty)

        debug = {
            "topic_avg":   round(topic_avg,  4),
            "topic_min":   round(topic_min,  4),
            "coherence":   round(coherence,  4),
            "off_ratio":   round(off_ratio,  4),
            "penalty":     round(penalty,    4),
            "dyn_sim_min": round(dyn_min,    4),
            "dyn_sim_max": round(dyn_max,    4),
        }
        return score_3, debug

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def topic_similarity(self, anchor: str, doc: str) -> float:
        """
        anchor와 doc 전체 텍스트의 코사인 유사도.
        단독 사용 또는 디버그용으로 유지.
        predict() 내에서는 _encode_sentences()를 사용하므로 이 메서드는 호출되지 않음.
        """
        a = self.topic_model.encode([anchor], normalize_embeddings=True, convert_to_numpy=True)[0]
        d = self.topic_model.encode([doc],    normalize_embeddings=True, convert_to_numpy=True)[0]
        return float(a @ d)

    def sentence_evidence(
        self,
        anchor: str,
        doc: str,
        abs_off_th: float = ABS_OFF_TH,
        gap_th: float = GAP_TH,
        top_k_on: int = 1,
        top_k_off: int = 1,
        *,
        _sents: Optional[List[str]] = None,
        _sims: Optional[np.ndarray] = None,
    ) -> Dict[str, List[str]]:
        """
        주제와 가장 가까운 문장(on_topic)과 가장 먼 문장(off_topic)을 추출.

        _sents, _sims가 주어지면 내부 인코딩 생략 (predict()에서 사전 계산값 재사용).

        off_topic 판정 조건 (둘 다 만족해야 표시):
          1. 절대 유사도 < ABS_OFF_TH(0.35)
          2. best_sim - worst_sim > GAP_TH(0.20)
             → 전체가 다 낮으면 off_topic 문장이 따로 없는 것
        """
        if _sents is not None and _sims is not None:
            sents, sims = _sents, _sims
        else:
            sents, _, _, sims = self._encode_sentences(anchor, doc)

        if len(sents) == 0:
            return {"on_topic_sentences": [], "off_topic_sentences": []}
        if len(sents) == 1:
            return {"on_topic_sentences": sents, "off_topic_sentences": []}

        order_desc = np.argsort(-sims)
        order_asc  = np.argsort(sims)
        best_sim   = float(sims[int(order_desc[0])])
        worst_sim  = float(sims[int(order_asc[0])])

        on_sents  = [sents[int(i)] for i in order_desc[:top_k_on]]
        off_sents: List[str] = []
        if worst_sim < abs_off_th and (best_sim - worst_sim) > gap_th:
            off_sents = [sents[int(i)] for i in order_asc[:top_k_off]]

        return {"on_topic_sentences": on_sents, "off_topic_sentences": off_sents}

    def sentence_analysis(
        self,
        anchor: str,
        doc: str,
        off_topic_th: float = ABS_OFF_TH,
        coherence_th: float = 0.40,
        quality_th: int = 35,
        *,
        _sents: Optional[List[str]] = None,
        _sent_vecs: Optional[np.ndarray] = None,
        _sims: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        문장별 상세 분석. LLM 피드백 생성 시 입력 데이터로 활용 예정.

        _sents, _sent_vecs, _sims가 주어지면 내부 인코딩 생략 (predict()에서 재사용).
        품질 점수는 _batch_quality_scores()로 배치 처리 (순차 호출 대비 속도 개선).

        각 문장마다 계산:
          - topic_score   : anchor cosine → sim_to_topic_score() → 0~100
          - coherence_sim : 인접 문장과의 평균 유사도
          - quality_score : label_model 확률 가중합 → 0~100
          - flags         : ["off_topic", "low_coherence", "low_quality"]
        """
        if _sents is not None and _sent_vecs is not None and _sims is not None:
            sents, sent_vecs, topic_sims = _sents, _sent_vecs, list(_sims)
        else:
            sents, sent_vecs, _, sims_arr = self._encode_sentences(anchor, doc)
            topic_sims = list(sims_arr) if len(sims_arr) > 0 else []

        if not sents:
            return []

        # ── 품질 점수 배치 계산 (문장마다 개별 호출 → 1회 배치로 대체) ────
        q_raws = self._batch_quality_scores(sents)   # List[float], 0~2

        results = []
        for i, (sent, tvec) in enumerate(zip(sents, sent_vecs)):
            tsim        = topic_sims[i]
            topic_score = score3_to_100(sim_to_topic_score(tsim))

            # 인접 문장과의 평균 유사도 (흐름 연속성 측정)
            neighbors = []
            if i > 0:
                neighbors.append(float(sent_vecs[i - 1] @ tvec))
            if i < len(sents) - 1:
                neighbors.append(float(sent_vecs[i + 1] @ tvec))
            coherence_sim = float(np.mean(neighbors)) if neighbors else None

            quality_score = int(min(100, round(q_raws[i] * 50)))

            flags = []
            if tsim < off_topic_th:
                flags.append("off_topic")
            if coherence_sim is not None and coherence_sim < coherence_th:
                flags.append("low_coherence")
            if quality_score < quality_th:
                flags.append("low_quality")

            results.append({
                "sentence":      sent,
                "topic_score":   topic_score,
                "coherence_sim": round(coherence_sim, 3) if coherence_sim is not None else None,
                "quality_score": quality_score,
                "flags":         flags,
            })

        return results

    def predict(
        self,
        topic_summary: str,
        topic_desc: str,
        topic_tags: List[str],
        doc_text: str,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        채점 메인 함수. /score 엔드포인트에서 호출.

        처리 순서:
          1. anchor 텍스트 생성
          2. _encode_sentences(): 문장 분리 + 임베딩 1회 수행
          3. _aggregate_topic(): 문장별 sim 집계 → topic_score_3
          4. quality_score(): 전체 문서 품질 → q_3
          5. 가중평균(ALPHA=0.35) → final_100
          6. on_topic 판정: topic_avg < T_OFF_SIM → final=0 강제
          7. sentence_evidence / sentence_analysis (사전 계산 벡터 재사용)
          8. worst_sentence: topic_score 최하위 문장 추출

        반환 구조:
          on_topic          : 주제 적합 여부 (bool)
          scores            : { final, topic, quality } (0~100)
          keywords          : TF-IDF 핵심 키워드 5개
          evidence          : { on_topic_sentences, off_topic_sentences }
          sentence_analysis : 문장별 분석 리스트
          worst_sentence    : topic_score 최하위 문장 (LLM 피드백용)
          debug             : 내부 계산값 (debug=True일 때만 포함)
        """
        anchor = build_anchor_text(topic_summary, topic_desc, topic_tags)

        # ── 문장 분리 + 임베딩 (1회만 수행) ──────────────────────────────────
        sents, sent_vecs, anchor_vec, sims = self._encode_sentences(anchor, doc_text)

        # 빈 텍스트 early return
        if not sents:
            return {
                "on_topic": False,
                "scores":   {"final": 0, "topic": 0, "quality": 0},
                "keywords": [],
                "evidence": {"on_topic_sentences": [], "off_topic_sentences": []},
                "sentence_analysis": [],
                "worst_sentence":    None,
            }

        # ── topic score (문장 수준 집계) ──────────────────────────────────────
        topic_score_3, agg_debug = self._aggregate_topic(sims, sent_vecs)

        # ── quality score (전체 문서 기준) ────────────────────────────────────
        q_2     = self.quality_score(doc_text)   # 0~2
        q_3     = q_2 * 1.5                       # 0~3 스케일 맞춤

        # ── 가중평균 ──────────────────────────────────────────────────────────
        final_3 = ALPHA * topic_score_3 + (1.0 - ALPHA) * q_3

        # ── on/off topic 판정 (3가지 조건 AND) ──────────────────────────────
        # KoSimCSE cosine baseline 문제로 인한 false positive 방지:
        #   단일 threshold로는 무관한 텍스트도 0.30+ 나와 통과 가능
        #   → avg / min / off_ratio 세 조건 모두 만족해야 on_topic
        topic_avg = agg_debug.get("topic_avg", float(np.mean(sims)))
        topic_min = agg_debug.get("topic_min", float(np.min(sims)))
        off_ratio = agg_debug.get("off_ratio", float(np.mean(sims < PENALTY_TH)))

        cond_avg      = topic_avg >= T_OFF_SIM
        cond_min      = topic_min >= MIN_SIM_TH
        cond_off_ratio = off_ratio < OFF_RATIO_TH
        on_topic      = cond_avg and cond_min and cond_off_ratio

        topic_100   = score3_to_100(topic_score_3)
        quality_100 = int(min(100, round(q_2 * 50)))
        final_100   = score3_to_100(final_3)
        if not on_topic:
            final_100 = 0

        # ── 부가 정보 생성 (사전 계산 벡터 재사용) ───────────────────────────
        if on_topic and topic_100 > 45:
            evidence = self.sentence_evidence(
                anchor, doc_text, _sents=sents, _sims=sims
            )
        else:
            # 전체 주제 이탈 시 모든 문장을 off_topic으로 표시
            evidence = {"on_topic_sentences": [], "off_topic_sentences": sents}

        sent_analysis = self.sentence_analysis(
            anchor, doc_text,
            _sents=sents, _sent_vecs=sent_vecs, _sims=sims,
        )

        # ── worst_sentence: topic_score 최하위 문장 (LLM 피드백용) ───────────
        worst_sentence = None
        if sent_analysis:
            worst = min(sent_analysis, key=lambda x: x["topic_score"])
            worst_sentence = {
                "sentence":    worst["sentence"],
                "topic_score": worst["topic_score"],
                "flags":       worst["flags"],
            }

        result: Dict[str, Any] = {
            "on_topic": on_topic,
            "scores": {
                "final":   final_100,
                "topic":   topic_100,
                "quality": quality_100,
            },
            "keywords":          extract_keywords(doc_text, top_n=5),
            "evidence":          evidence,
            "sentence_analysis": sent_analysis,
            "worst_sentence":    worst_sentence,
        }

        if debug:
            result["debug"] = {
                **agg_debug,
                "quality_class":   int(round(q_2)),
                "quality_score_2": round(q_2, 2),
                "topic_score_3":   round(topic_score_3, 2),
                "final_score_3":   round(final_3, 2),
                "on_topic_reason": {
                    "cond_avg":       f"topic_avg({topic_avg:.3f}) >= T_OFF_SIM({T_OFF_SIM}) → {cond_avg}",
                    "cond_min":       f"topic_min({topic_min:.3f}) >= MIN_SIM_TH({MIN_SIM_TH}) → {cond_min}",
                    "cond_off_ratio": f"off_ratio({off_ratio:.3f}) < OFF_RATIO_TH({OFF_RATIO_TH}) → {cond_off_ratio}",
                    "result":         on_topic,
                },
                "calib": {
                    "T_OFF_SIM":    T_OFF_SIM,
                    "MIN_SIM_TH":   MIN_SIM_TH,
                    "OFF_RATIO_TH": OFF_RATIO_TH,
                    "ALPHA":        ALPHA,
                    "W_AVG":        W_AVG,
                    "W_MIN":        W_MIN,
                    "W_COH":        W_COH,
                    "PENALTY_TH":   PENALTY_TH,
                    "PENALTY_W":    PENALTY_W,
                    "ABS_OFF_TH":   ABS_OFF_TH,
                    "GAP_TH":       GAP_TH,
                }
            }

        return result


# ── 단독 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = ServiceScorer()

    result = scorer.predict(
        topic_summary="연애의 어려움",
        topic_desc="이성 관계에서 겪는 고민과 감정",
        topic_tags=["#연애", "#이성", "#감정"],
        doc_text="오늘 도서관에서 경제학 책을 빌렸습니다. GDP와 환율의 상관관계가 흥미로웠습니다.",
        debug=True
    )

    print(result)