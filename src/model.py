import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from .preprocess import build_prompts, extract_final_number, normalize_answer


@dataclass
class GenerationResult:
    text: str
    score: float
    tokens: int


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    if n == 0 or k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        cdf = 0.0
        for i in range(k + 1):
            cdf += math.comb(n, i) * (mid**i) * ((1 - mid) ** (n - i))
        if cdf <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def conformal_pvalue_ge(cal_scores: List[float], s: float) -> float:
    ge = sum(1 for cs in cal_scores if cs >= s)
    return (1.0 + ge) / (len(cal_scores) + 1.0)


class GenerationHelper:
    def __init__(self, model, tokenizer, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    @torch.no_grad()
    def generate_text_score_tokens(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.cfg.model.generation.do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        seq = out.sequences[0]
        text = self.tokenizer.decode(seq, skip_special_tokens=True)
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = seq[in_len:]

        nlls, ents = [], []
        for t, step_logits in enumerate(out.scores):
            logits = step_logits[0]
            logp = torch.log_softmax(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            chosen = int(gen_ids[t]) if t < len(gen_ids) else int(torch.argmax(logits))
            nlls.append(float(-logp[chosen].item()))
            ents.append(float(-(p * logp).sum().item()))

        mean_nll = float(np.mean(nlls)) if nlls else 0.0
        mean_ent = float(np.mean(ents)) if ents else 0.0
        s = mean_nll + float(self.cfg.controller.entropy_weight) * mean_ent
        gen_len = int(out.sequences.shape[-1] - in_len)
        return GenerationResult(text=text, score=s, tokens=gen_len)


class CAMCoTController:
    def __init__(self, cfg, gen: GenerationHelper):
        self.cfg = cfg
        self.gen = gen
        self.tau1 = None
        self.tau2 = None
        self.cal_scores = None

    def calibrate(self, cal_examples) -> Dict[str, Any]:
        s1s, ok1s, t1s = [], [], []
        s2s, ok2s, t2s = [], [], []
        t3s = []
        for ex in cal_examples:
            q = ex["question"]
            gold = normalize_answer(ex["answer"], ex.get("normalized_answer"))
            prompts = build_prompts(q)
            r1 = self.gen.generate_text_score_tokens(prompts["direct"], self.cfg.model.generation.max_new_tokens.direct)
            r2 = self.gen.generate_text_score_tokens(prompts["brief"], self.cfg.model.generation.max_new_tokens.brief)
            r3 = self.gen.generate_text_score_tokens(prompts["full"], self.cfg.model.generation.max_new_tokens.full)

            pred1 = extract_final_number(r1.text, self.cfg.dataset.preprocessing.extract_answer_regex, self.cfg.dataset.preprocessing.normalize_commas)
            pred2 = extract_final_number(r2.text, self.cfg.dataset.preprocessing.extract_answer_regex, self.cfg.dataset.preprocessing.normalize_commas)
            ok1 = pred1 is not None and pred1 == gold
            ok2 = pred2 is not None and pred2 == gold

            s1s.append(r1.score)
            ok1s.append(ok1)
            t1s.append(r1.tokens)
            s2s.append(r2.score)
            ok2s.append(ok2)
            t2s.append(r2.tokens)
            t3s.append(r3.tokens)

        s1s = np.array(s1s)
        s2s = np.array(s2s)
        ok1s = np.array(ok1s, dtype=bool)
        ok2s = np.array(ok2s, dtype=bool)
        t1s = np.array(t1s)
        t2s = np.array(t2s)
        t3s = np.array(t3s)

        q_grid = np.linspace(0.05, 0.95, int(self.cfg.controller.threshold_grid.quantiles))
        c1 = np.quantile(s1s, q_grid)
        c2 = np.quantile(s2s, q_grid)

        best = None
        forced_flags = np.array(
            [conformal_pvalue_ge(s1s.tolist(), float(s)) < self.cfg.controller.p_min_ood for s in s1s]
        )
        for tau1 in sorted(set(map(float, c1))):
            forced = forced_flags if self.cfg.controller.ood_guard else np.zeros_like(s1s, dtype=bool)
            A1 = (s1s <= tau1) & (~forced)
            n1 = int(A1.sum())
            k1 = int((~ok1s[A1]).sum()) if n1 else 0
            U1 = clopper_pearson_upper(k1, n1, self.cfg.controller.delta)
            if U1 > self.cfg.controller.alpha1:
                continue
            for tau2 in sorted(set(map(float, c2))):
                not1 = ~A1
                A2 = not1 & (s2s <= tau2) & (~forced)
                n2 = int(A2.sum())
                k2 = int((~ok2s[A2]).sum()) if n2 else 0
                U2 = clopper_pearson_upper(k2, n2, self.cfg.controller.delta)
                if U2 > self.cfg.controller.alpha2:
                    continue
                cost = (A1 * t1s + A2 * (t1s + t2s) + (~(A1 | A2)) * (t1s + t2s + t3s)).mean()
                key = (cost, -(A1.mean() + A2.mean()))
                if best is None or key < best[0]:
                    best = (key, tau1, tau2, U1, U2, n1, n2, float(A1.mean()), float(A2.mean()))

        if best is None:
            self.tau1 = float(np.min(s1s))
            self.tau2 = float(np.min(s2s))
            U1 = U2 = 1.0
            n1 = n2 = 0
            cov1 = cov2 = 0.0
        else:
            _, self.tau1, self.tau2, U1, U2, n1, n2, cov1, cov2 = best
        self.cal_scores = s1s.tolist()
        return {
            "tau1": self.tau1,
            "tau2": self.tau2,
            "U1": U1,
            "U2": U2,
            "n1": n1,
            "n2": n2,
            "cov1": cov1,
            "cov2": cov2,
            "cal_scores": self.cal_scores,
        }

    def solve(self, question: str) -> Dict[str, Any]:
        prompts = build_prompts(question)
        r1 = self.gen.generate_text_score_tokens(prompts["direct"], self.cfg.model.generation.max_new_tokens.direct)
        forced = False
        if self.cfg.controller.ood_guard:
            p = conformal_pvalue_ge(self.cal_scores, r1.score)
            forced = p < self.cfg.controller.p_min_ood
        if (r1.score <= self.tau1) and (not forced):
            return {"text": r1.text, "score": r1.score, "tokens": r1.tokens, "route": "direct", "forced": forced, "s1": r1.score}
        r2 = self.gen.generate_text_score_tokens(prompts["brief"], self.cfg.model.generation.max_new_tokens.brief)
        if (r2.score <= self.tau2) and (not forced):
            return {
                "text": r2.text,
                "score": r2.score,
                "tokens": r1.tokens + r2.tokens,
                "route": "brief",
                "forced": forced,
                "s1": r1.score,
                "s2": r2.score,
            }
        r3 = self.gen.generate_text_score_tokens(prompts["full"], self.cfg.model.generation.max_new_tokens.full)
        return {
            "text": r3.text,
            "score": r3.score,
            "tokens": r1.tokens + r2.tokens + r3.tokens,
            "route": "full",
            "forced": forced,
            "s1": r1.score,
            "s2": r2.score,
        }


class CRACoTController:
    def __init__(self, cfg, gen: GenerationHelper):
        self.cfg = cfg
        self.gen = gen
        self.tau = None
        self.cal_scores = None

    def calibrate(self, cal_examples) -> Dict[str, Any]:
        s1s, ok1s = [], []
        for ex in cal_examples:
            q = ex["question"]
            gold = normalize_answer(ex["answer"], ex.get("normalized_answer"))
            prompts = build_prompts(q)
            r1 = self.gen.generate_text_score_tokens(prompts["direct"], self.cfg.model.generation.max_new_tokens.direct)
            pred1 = extract_final_number(r1.text, self.cfg.dataset.preprocessing.extract_answer_regex, self.cfg.dataset.preprocessing.normalize_commas)
            ok1 = pred1 is not None and pred1 == gold
            s1s.append(r1.score)
            ok1s.append(ok1)
        s1s = np.array(s1s)
        ok1s = np.array(ok1s, dtype=bool)

        q_grid = np.linspace(0.05, 0.95, int(self.cfg.controller.threshold_grid.quantiles))
        cands = np.quantile(s1s, q_grid)
        tau = float(np.min(s1s))
        best_U1 = 1.0
        best_n1 = 0
        for cand in sorted(set(map(float, cands))):
            idx = s1s <= cand
            n1 = int(idx.sum())
            k1 = int((~ok1s[idx]).sum()) if n1 else 0
            U1 = clopper_pearson_upper(k1, n1, self.cfg.controller.delta)
            if U1 <= self.cfg.controller.alpha1:
                tau = cand
                best_U1 = U1
                best_n1 = n1
        self.tau = tau
        self.cal_scores = s1s.tolist()
        return {"tau": self.tau, "U1": best_U1, "n1": best_n1, "cal_scores": self.cal_scores}

    def solve(self, question: str) -> Dict[str, Any]:
        prompts = build_prompts(question)
        r1 = self.gen.generate_text_score_tokens(prompts["direct"], self.cfg.model.generation.max_new_tokens.direct)
        forced = False
        if self.cfg.controller.ood_guard:
            p = conformal_pvalue_ge(self.cal_scores, r1.score)
            forced = p < self.cfg.controller.p_min_ood
        if (r1.score <= self.tau) and (not forced):
            return {"text": r1.text, "score": r1.score, "tokens": r1.tokens, "route": "direct", "forced": forced, "s1": r1.score}
        r3 = self.gen.generate_text_score_tokens(prompts["full"], self.cfg.model.generation.max_new_tokens.full)
        return {
            "text": r3.text,
            "score": r3.score,
            "tokens": r1.tokens + r3.tokens,
            "route": "full",
            "forced": forced,
            "s1": r1.score,
        }
