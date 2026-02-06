import copy
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_str: Optional[str]) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_str is None:
        return torch.float32
    key = str(dtype_str).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[key]


def normalize_number_str(num_str: str) -> str:
    return num_str.replace(",", "")


def parse_number(num_str: Optional[str]) -> Optional[float]:
    if num_str is None:
        return None
    try:
        return float(normalize_number_str(num_str))
    except (ValueError, TypeError):
        return None


def canonical_number(num_str: Optional[str]) -> Optional[str]:
    val = parse_number(num_str)
    if val is None:
        return None
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.6f}".rstrip("0").rstrip(".")


def extract_numbers(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    numbers = NUM_RE.findall(text)
    return [n for n in (canonical_number(n) for n in numbers) if n is not None]


def extract_final_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text_clean = text.replace(",", "")
    lower = text_clean.lower()
    markers = ["answer:", "answer is", "final answer", "so the answer", "thus the answer"]
    idx = -1
    for marker in markers:
        pos = lower.rfind(marker)
        if pos > idx:
            idx = pos + len(marker)
    substr = text_clean[idx:] if idx != -1 else text_clean
    nums = extract_numbers(substr)
    if nums:
        return nums[-1]
    nums = extract_numbers(text_clean)
    return nums[-1] if nums else None


def numbers_preserved(q1: str, q2: str) -> bool:
    return Counter(extract_numbers(q1)) == Counter(extract_numbers(q2))


def agreement_rate(answers: List[Optional[str]]) -> Tuple[float, Optional[str]]:
    clean = [a for a in answers if a is not None]
    if not clean:
        return 0.0, None
    counts = Counter(clean)
    maj, ct = counts.most_common(1)[0]
    return ct / len(clean), maj


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def compare_numbers(pred: Optional[str], gold: Optional[str], tol: float = 1e-3) -> bool:
    if pred is None or gold is None:
        return False
    pred_val = parse_number(pred)
    gold_val = parse_number(gold)
    if pred_val is None or gold_val is None:
        return False
    if abs(pred_val - round(pred_val)) < 1e-6 and abs(gold_val - round(gold_val)) < 1e-6:
        return int(round(pred_val)) == int(round(gold_val))
    return abs(pred_val - gold_val) <= tol


def _clone_cfg(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return copy.deepcopy(cfg)


def update_cfg_from_optuna_params(run_cfg: Any, params: Dict[str, Any]) -> Any:
    cfg = _clone_cfg(run_cfg)
    if "tau" in params:
        cfg.method_config.tau_reliability = float(params["tau"])
    if "m" in params:
        cfg.method_config.m_self_consistency = int(params["m"])
    if "p" in params:
        cfg.method_config.p_paraphrases = int(params["p"])
    if "temperature_sc" in params:
        cfg.model.generation.sample_temperature_sc = float(params["temperature_sc"])
    if "rcc_weight_sim" in params and hasattr(cfg.method_config, "reconstruction"):
        cfg.method_config.reconstruction.rcc_weight_sim = float(params["rcc_weight_sim"])
        if hasattr(cfg.method_config.reconstruction, "rcc_weight_num_match"):
            cfg.method_config.reconstruction.rcc_weight_num_match = float(1.0 - params["rcc_weight_sim"])
    return cfg


class LLMWrapper:
    def __init__(self, model_cfg: Any, cache_dir: str) -> None:
        self.model_cfg = model_cfg
        config = AutoConfig.from_pretrained(
            model_cfg.hf_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_name,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        added_pad = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                added_pad = True
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is missing and cannot be set.")
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        dtype = resolve_dtype(model_cfg.dtype)
        model_cls = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        self.model = model_cls.from_pretrained(
            model_cfg.hf_name,
            torch_dtype=dtype,
            device_map="auto" if model_cfg.device == "cuda" else None,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if added_pad:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if model_cfg.device != "cuda":
            self.model.to(model_cfg.device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        assert self.tokenizer.pad_token_id is not None
        assert getattr(self.model.config, "vocab_size", 0) > 0

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    @torch.inference_mode()
    def generate_texts(
        self,
        prompts: List[str],
        n: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        max_length: Optional[int] = None,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if not do_sample:
            n = 1
        n = max(1, int(n))
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_lengths = batch["attention_mask"].sum(dim=1).tolist()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        gen_kwargs: Dict[str, Any] = {
            "do_sample": do_sample,
            "num_return_sequences": n,
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        outputs = self.model.generate(**batch, **gen_kwargs)
        if self.is_encoder_decoder:
            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [t.strip() for t in texts]

        total = outputs.shape[0]
        expected = len(prompts) * n
        if total != expected:
            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [t.strip() for t in texts]

        outputs = outputs.view(len(prompts), n, -1)
        decoded: List[str] = []
        for i in range(len(prompts)):
            gen_ids = outputs[i, :, input_lengths[i] :]
            decoded.extend(self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
        return [t.strip() for t in decoded]


class EmbeddingWrapper:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=32,
        )


class DemoBuilderBase:
    def __init__(self, llm: LLMWrapper, embedder: EmbeddingWrapper, run_cfg: Any) -> None:
        self.llm = llm
        self.embedder = embedder
        self.run_cfg = run_cfg
        self.method_cfg = run_cfg.method_config
        self.gen_cfg = run_cfg.model.generation
        self.max_length = int(run_cfg.dataset.preprocessing.max_length)

    def sample_cot(self, question: str, n_samples: int, temperature: float) -> List[str]:
        prompt = f"Q: {question}\nA: Let's think step by step."
        return self.llm.generate_texts(
            prompts=[prompt],
            n=n_samples,
            do_sample=True,
            temperature=temperature,
            top_p=float(self.gen_cfg.sample_top_p),
            max_new_tokens=int(self.run_cfg.model.max_new_tokens),
            max_length=int(self.max_length),
        )

    def paraphrase_question(self, question: str, n_para: int) -> List[str]:
        prompt = (
            "Paraphrase the following word problem without changing its meaning or numbers. "
            "Return only the paraphrased question.\n\n"
            f"Problem: {question}\nParaphrase:"
        )
        paras = self.llm.generate_texts(
            prompts=[prompt],
            n=n_para,
            do_sample=True,
            temperature=float(self.gen_cfg.paraphrase_temperature),
            top_p=float(self.gen_cfg.sample_top_p),
            max_new_tokens=128,
            max_length=int(self.max_length),
        )
        out: List[str] = []
        for p in paras:
            p_clean = p.strip()
            if not p_clean:
                continue
            if p_clean.lower() == question.strip().lower():
                continue
            if p_clean not in out:
                out.append(p_clean)
        return out[:n_para]

    def reconstruct_question(self, chain_text: str) -> str:
        prompt = (
            "You are given a step-by-step solution to a math word problem. "
            "Reconstruct the original word problem as precisely as possible, preserving all numbers. "
            "Return only the problem statement.\n\n"
            f"Solution:\n{chain_text}\n\nProblem:"
        )
        recon_cfg = getattr(self.method_cfg, "reconstruction", None)
        deterministic = True
        temperature = float(self.gen_cfg.sample_temperature_sc)
        top_p = float(self.gen_cfg.sample_top_p)
        if recon_cfg is not None:
            deterministic = bool(getattr(recon_cfg, "deterministic", True))
            temperature = float(getattr(recon_cfg, "temperature", temperature))
            top_p = float(getattr(recon_cfg, "top_p", top_p))
        return self.llm.generate_texts(
            prompts=[prompt],
            n=1,
            do_sample=not deterministic,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=128,
            max_length=int(self.max_length),
        )[0].strip()


class PIRAutoCoTBuilder(DemoBuilderBase):
    def build_demo(self, question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        paras_raw = self.paraphrase_question(question, int(self.method_cfg.p_paraphrases))
        paras = paras_raw
        if bool(self.method_cfg.number_preservation_filter):
            paras = [p for p in paras_raw if numbers_preserved(question, p)]
        if len(paras) == 0:
            return None

        variants = [question] + paras
        variant_answers: List[List[Optional[str]]] = []
        variant_chains: List[List[str]] = []
        all_answers: List[Optional[str]] = []

        for qv in variants:
            chains = self.sample_cot(
                qv,
                int(self.method_cfg.m_self_consistency),
                float(self.gen_cfg.sample_temperature_sc),
            )
            answers = [extract_final_number(c) for c in chains]
            variant_chains.append(chains)
            variant_answers.append(answers)
            all_answers.extend(answers)

        _, global_maj = agreement_rate(all_answers)
        if global_maj is None:
            return None

        r_sc, _ = agreement_rate(variant_answers[0])
        pi_scores = []
        for answers in variant_answers[1:]:
            valid = [a for a in answers if a is not None]
            if valid:
                pi_scores.append(sum(a == global_maj for a in valid) / len(valid))
            else:
                pi_scores.append(0.0)
        r_pi = float(np.mean(pi_scores)) if pi_scores else 0.0
        r = float(r_sc * r_pi)

        if r < float(self.method_cfg.tau_reliability):
            return None

        rep_chain = None
        for chain in variant_chains[0]:
            if extract_final_number(chain) == global_maj:
                rep_chain = chain
                break
        if rep_chain is None:
            return None

        demo = f"Q: {question}\nA: {rep_chain}\n"
        diag = {
            "r_sc": float(r_sc),
            "r_pi": float(r_pi),
            "r_cc": None,
            "r": float(r),
            "global_maj": global_maj,
            "n_paraphrases_generated": len(paras_raw),
            "n_paraphrases_kept": len(paras),
        }
        return demo, diag

    def build_demos(
        self,
        questions: List[str],
        embeddings: np.ndarray,
        kmeans: KMeans,
        log_fn=None,
        log_step_start: int = 0,
    ) -> Tuple[List[str], List[Dict[str, Any]], int]:
        return _build_demos_for_clusters(
            builder=self,
            questions=questions,
            embeddings=embeddings,
            kmeans=kmeans,
            log_fn=log_fn,
            log_step_start=log_step_start,
        )


class C3AutoCoTBuilder(DemoBuilderBase):
    def build_demo(self, question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        paras_raw = self.paraphrase_question(question, int(self.method_cfg.p_paraphrases))
        paras = paras_raw
        if bool(self.method_cfg.number_preservation_filter):
            paras = [p for p in paras_raw if numbers_preserved(question, p)]
        if len(paras) == 0:
            return None

        variants = [question] + paras
        variant_answers: List[List[Optional[str]]] = []
        variant_chains: List[List[str]] = []
        all_answers: List[Optional[str]] = []

        for qv in variants:
            chains = self.sample_cot(
                qv,
                int(self.method_cfg.m_self_consistency),
                float(self.gen_cfg.sample_temperature_sc),
            )
            answers = [extract_final_number(c) for c in chains]
            variant_chains.append(chains)
            variant_answers.append(answers)
            all_answers.extend(answers)

        _, global_maj = agreement_rate(all_answers)
        if global_maj is None:
            return None

        r_sc, _ = agreement_rate(variant_answers[0])
        pi_scores = []
        for answers in variant_answers[1:]:
            valid = [a for a in answers if a is not None]
            if valid:
                pi_scores.append(sum(a == global_maj for a in valid) / len(valid))
            else:
                pi_scores.append(0.0)
        r_pi = float(np.mean(pi_scores)) if pi_scores else 0.0

        rep_chain = None
        for chain in variant_chains[0]:
            if extract_final_number(chain) == global_maj:
                rep_chain = chain
                break
        if rep_chain is None:
            return None

        q_hat = self.reconstruct_question(rep_chain)
        emb_q = self.embedder.encode([question], normalize=True)[0]
        emb_hat = self.embedder.encode([q_hat], normalize=True)[0]
        sim = cosine_sim(emb_q, emb_hat)
        num_ok = 1.0 if numbers_preserved(question, q_hat) else 0.0
        w_sim = float(self.method_cfg.reconstruction.rcc_weight_sim)
        w_num = float(self.method_cfg.reconstruction.rcc_weight_num_match)
        weight_sum = w_sim + w_num
        if weight_sum <= 0:
            w_sim, w_num = 0.5, 0.5
        else:
            w_sim, w_num = w_sim / weight_sum, w_num / weight_sum
        r_cc = w_sim * sim + w_num * num_ok

        r = float(r_sc * r_pi * r_cc)
        if r < float(self.method_cfg.tau_reliability):
            return None

        demo = f"Q: {question}\nA: {rep_chain}\n"
        diag = {
            "r_sc": float(r_sc),
            "r_pi": float(r_pi),
            "r_cc": float(r_cc),
            "r": float(r),
            "global_maj": global_maj,
            "n_paraphrases_generated": len(paras_raw),
            "n_paraphrases_kept": len(paras),
        }
        return demo, diag

    def build_demos(
        self,
        questions: List[str],
        embeddings: np.ndarray,
        kmeans: KMeans,
        log_fn=None,
        log_step_start: int = 0,
    ) -> Tuple[List[str], List[Dict[str, Any]], int]:
        return _build_demos_for_clusters(
            builder=self,
            questions=questions,
            embeddings=embeddings,
            kmeans=kmeans,
            log_fn=log_fn,
            log_step_start=log_step_start,
        )


def _diagnostic_flags(method_cfg: Any) -> Dict[str, bool]:
    diag_cfg = getattr(method_cfg, "diagnostics", None)
    return {
        "track_acceptance_rate": bool(getattr(diag_cfg, "track_acceptance_rate", False))
        if diag_cfg is not None
        else False,
        "track_reliability_components": bool(getattr(diag_cfg, "track_reliability_components", False))
        if diag_cfg is not None
        else False,
    }


def _resolve_fallback_mode(method_cfg: Any) -> str:
    mode = str(getattr(method_cfg, "fallback_demo", "single_chain_if_rejected")).lower()
    if mode in {"single_chain_if_rejected", "single_chain", "fallback"}:
        return "single_chain"
    if mode in {"none", "skip", "skip_cluster", "no_demo"}:
        return "none"
    return "single_chain"


def _build_demos_for_clusters(
    builder: DemoBuilderBase,
    questions: List[str],
    embeddings: np.ndarray,
    kmeans: KMeans,
    log_fn=None,
    log_step_start: int = 0,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    demos: List[str] = []
    stats: List[Dict[str, Any]] = []
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    step = log_step_start
    diag_flags = _diagnostic_flags(builder.method_cfg)
    fallback_mode = _resolve_fallback_mode(builder.method_cfg)

    for cluster_id in range(kmeans.n_clusters):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue
        dists = np.linalg.norm(embeddings[idxs] - centers[cluster_id], axis=1)
        order = idxs[np.argsort(dists)]

        accepted = False
        candidates_tried = 0
        demo_text = None
        diag: Optional[Dict[str, Any]] = None

        for rank, idx in enumerate(order[: int(builder.method_cfg.max_candidates_per_cluster)]):
            candidates_tried += 1
            res = builder.build_demo(questions[idx])
            if res is not None:
                demo_text, diag = res
                accepted = True
                break
            if log_fn is not None and diag_flags["track_acceptance_rate"]:
                log_fn(
                    {
                        "demo_cluster": cluster_id,
                        "demo_candidate_rank": int(rank),
                        "accepted": 0,
                    },
                    step=step,
                )
                step += 1

        if not accepted or demo_text is None or diag is None:
            diag = {
                "r_sc": 0.0,
                "r_pi": 0.0,
                "r_cc": 0.0,
                "r": 0.0,
                "global_maj": None,
                "n_paraphrases_generated": 0,
                "n_paraphrases_kept": 0,
            }
            if fallback_mode == "single_chain":
                fallback_q = questions[order[0]]
                chain = builder.sample_cot(
                    fallback_q,
                    1,
                    float(builder.gen_cfg.sample_temperature_sc),
                )[0]
                demo_text = f"Q: {fallback_q}\nA: {chain}\n"
            else:
                demo_text = ""

        demos.append(demo_text)
        stat = {
            "cluster_id": cluster_id,
            "accepted": bool(accepted),
            "candidates_tried": int(candidates_tried),
            **diag,
        }
        stats.append(stat)

        if log_fn is not None:
            log_payload = {
                "demo_cluster": cluster_id,
                "demo_candidate_rank": int(candidates_tried - 1),
                "accepted": int(accepted),
            }
            if diag_flags["track_reliability_components"]:
                log_payload.update(
                    {
                        "r_sc": stat["r_sc"],
                        "r_pi": stat["r_pi"],
                        "r_cc": stat["r_cc"],
                        "r": stat["r"],
                        "n_paraphrases_generated": stat["n_paraphrases_generated"],
                        "n_paraphrases_kept": stat["n_paraphrases_kept"],
                    }
                )
            if log_payload:
                log_fn(log_payload, step=step)
                step += 1

    return demos, stats, step


def build_prompt(demos: List[str], question: str) -> str:
    return "".join(demos) + f"Q: {question}\nA: Let's think step by step."


def predict_with_demos(
    llm: LLMWrapper,
    demos: List[str],
    question: str,
    max_new_tokens: int,
    max_length: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_prompt(demos, question)
    return llm.generate_texts(
        prompts=[prompt],
        n=1,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )[0]


def _assert_batch_shapes(llm: LLMWrapper, prompt: str, label: str, max_length: int) -> None:
    tok_inp = llm.tokenize([prompt], max_length=max_length)
    tok_lbl = llm.tokenize([label], max_length=max_length)
    assert tok_inp["input_ids"].shape[0] == tok_lbl["input_ids"].shape[0] == 1
    assert tok_inp["input_ids"].shape == tok_inp["attention_mask"].shape
    assert tok_lbl["input_ids"].shape == tok_lbl["attention_mask"].shape
    assert tok_inp["input_ids"].shape[1] > 0 and tok_lbl["input_ids"].shape[1] > 0


def evaluate_accuracy(
    llm: LLMWrapper,
    demos: List[str],
    test_examples: List[Dict[str, str]],
    max_new_tokens: int,
    max_length: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    log_fn=None,
    log_step_start: int = 0,
    assert_batch_shapes: bool = False,
) -> Tuple[float, List[Optional[str]], List[Optional[str]], List[int]]:
    correct = 0
    preds: List[Optional[str]] = []
    golds: List[Optional[str]] = []
    corrects: List[int] = []
    step = log_step_start

    for idx, ex in enumerate(test_examples):
        prompt = build_prompt(demos, ex["question"])
        if idx == 0 and assert_batch_shapes:
            _assert_batch_shapes(llm, prompt, str(ex["answer"]), max_length)

        response = llm.generate_texts(
            prompts=[prompt],
            n=1,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )[0]
        pred = extract_final_number(response)
        gold = extract_final_number(str(ex["answer"]))
        is_correct = int(compare_numbers(pred, gold))
        correct += is_correct
        preds.append(pred)
        golds.append(gold)
        corrects.append(is_correct)

        if log_fn is not None:
            error_val = None
            pred_val = parse_number(pred) if pred is not None else None
            gold_val = parse_number(gold) if gold is not None else None
            if pred_val is not None and gold_val is not None:
                error_val = pred_val - gold_val
            log_fn(
                {
                    "eval_step": idx,
                    "eval_pred": float(pred_val) if pred_val is not None else None,
                    "eval_gold": float(gold_val) if gold_val is not None else None,
                    "eval_error": error_val,
                    "eval_correct": is_correct,
                    "eval_accuracy_running": correct / (idx + 1),
                },
                step=step,
            )
            step += 1

    accuracy = correct / max(1, len(test_examples))
    return float(accuracy), preds, golds, corrects


def assign_clusters(test_embeddings: np.ndarray, kmeans: KMeans) -> np.ndarray:
    distances = kmeans.transform(test_embeddings)
    return np.argmin(distances, axis=1)


def compute_grounding_utility_correlation(
    demo_stats: List[Dict[str, Any]],
    test_assignments: np.ndarray,
    corrects: List[int],
    n_clusters: int,
) -> Dict[str, float]:
    cluster_correct = {i: [] for i in range(n_clusters)}
    for idx, cluster_id in enumerate(test_assignments):
        cluster_correct[int(cluster_id)].append(corrects[idx])
    cluster_acc = {
        cid: (float(np.mean(vals)) if len(vals) > 0 else None) for cid, vals in cluster_correct.items()
    }

    r_cc_vals = []
    util_vals = []
    for stat in demo_stats:
        r_cc = stat.get("r_cc")
        util = cluster_acc.get(stat["cluster_id"])
        if r_cc is None or util is None:
            continue
        r_cc_vals.append(float(r_cc))
        util_vals.append(float(util))

    if len(r_cc_vals) < 2:
        return {"grounding_utility_correlation": 0.0, "grounding_utility_spearman": 0.0}
    pearson = float(np.corrcoef(r_cc_vals, util_vals)[0, 1])
    spearman = float(spearmanr(r_cc_vals, util_vals).correlation)
    if np.isnan(pearson):
        pearson = 0.0
    if np.isnan(spearman):
        spearman = 0.0
    return {"grounding_utility_correlation": pearson, "grounding_utility_spearman": spearman}
