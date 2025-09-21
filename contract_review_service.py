"""Contract review service using heuristic analysis.

This module provides a small AI-inspired contract review helper that inspects a
contract body and highlights potential risk areas.  It relies on lightweight
natural language processing techniques such as keyword spotting, pattern
matching and heuristic scoring to approximate the behaviour of a real AI
contract reviewer without external model dependencies.

Example
-------
>>> from contract_review_service import ContractReviewService
>>> service = ContractReviewService()
>>> review = service.review('''
... The Supplier will deliver services within 30 days. Either party may
... terminate the agreement with 30 days' written notice. The Supplier shall
... maintain confidentiality of all proprietary data.''')
>>> print(service.generate_report(review))

The :func:`ContractReviewService.generate_report` method formats a
human-readable summary that lists risky clauses, strengths and recommended
actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Pattern, Sequence, Tuple


@dataclass(frozen=True)
class ClauseConfig:
    """Configuration describing how to evaluate a contract clause."""

    name: str
    keywords: Sequence[str]
    missing_risk: str
    summary: str
    recommendation: str
    warning_keywords: Sequence[str] = field(default_factory=tuple)
    positive_keywords: Sequence[str] = field(default_factory=tuple)


@dataclass
class ClauseResult:
    """Outcome of analysing a clause within the contract."""

    name: str
    present: bool
    risk_level: str
    matched_sentences: List[str]
    issues: List[str]
    notes: List[str]
    recommendation: str
    summary: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the result."""

        return {
            "name": self.name,
            "present": self.present,
            "risk_level": self.risk_level,
            "matched_sentences": list(self.matched_sentences),
            "issues": list(self.issues),
            "notes": list(self.notes),
            "recommendation": self.recommendation,
            "summary": self.summary,
        }


class ContractReviewService:
    """A lightweight AI-style contract review service.

    The service searches for essential clauses, evaluates their risk level based
    on heuristic keyword analysis and returns a structured report summarising
    the findings.
    """

    def __init__(self, clause_configs: Iterable[ClauseConfig] | None = None) -> None:
        self._clauses: List[ClauseConfig] = list(
            clause_configs if clause_configs is not None else self._default_configs()
        )

    @staticmethod
    def _default_configs() -> List[ClauseConfig]:
        """Return default clause configurations used by the service."""

        return [
            ClauseConfig(
                name="Termination",
                keywords=("terminate", "termination", "notice", "cancel"),
                missing_risk="high",
                summary="Termination 권한과 조건을 확인합니다.",
                recommendation="상호 합리적인 통지 기간과 사유를 명시하세요.",
                warning_keywords=("immediate", "sole discretion", "without cause"),
                positive_keywords=("written notice", "prior notice", "mutual"),
            ),
            ClauseConfig(
                name="Confidentiality",
                keywords=("confidential", "non-disclosure", "nda", "proprietary"),
                missing_risk="high",
                summary="기밀 유지 의무의 범위를 점검합니다.",
                recommendation="기밀 정보의 정의와 예외 사항을 명확히 하세요.",
                warning_keywords=("perpetual", "unlimited", "irrevocable"),
                positive_keywords=("return", "destroy", "survive"),
            ),
            ClauseConfig(
                name="Liability",
                keywords=("liability", "indemnify", "damages", "hold harmless"),
                missing_risk="high",
                summary="책임 및 손해배상 한도를 확인합니다.",
                recommendation="손해배상 한도와 면책 범위를 명확히 정의하세요.",
                warning_keywords=("unlimited", "all damages", "any damages"),
                positive_keywords=("cap", "limited", "maximum"),
            ),
            ClauseConfig(
                name="Payment Terms",
                keywords=("payment", "fee", "compensation", "invoice"),
                missing_risk="medium",
                summary="대금 지급 조건을 확인합니다.",
                recommendation="지급 시기와 조건, 지연 시 조치를 포함하세요.",
                warning_keywords=("late fee", "penalty", "interest"),
                positive_keywords=("net", "days", "schedule"),
            ),
            ClauseConfig(
                name="Intellectual Property",
                keywords=("intellectual property", "ip", "license", "ownership"),
                missing_risk="medium",
                summary="지식 재산권 귀속과 사용 조건을 확인합니다.",
                recommendation="귀속 주체와 사용 범위를 명시하세요.",
                warning_keywords=("assign", "transfer", "exclusive"),
                positive_keywords=("retain", "non-exclusive", "limited"),
            ),
            ClauseConfig(
                name="Governing Law",
                keywords=("governing law", "jurisdiction", "venue"),
                missing_risk="low",
                summary="준거법과 관할을 확인합니다.",
                recommendation="분쟁 해결 절차와 관할을 구체적으로 명시하세요.",
                warning_keywords=("exclusive jurisdiction", "foreign"),
                positive_keywords=("arbitration", "mediation"),
            ),
        ]

    def review(self, contract_text: str) -> Dict[str, object]:
        """Analyse the contract text and return a structured report."""

        normalised = self._normalise_text(contract_text)
        sentences = self._split_sentences(normalised)
        clause_results = [self._evaluate_clause(clause, sentences) for clause in self._clauses]
        overall_risk = self._calculate_overall_risk(clause_results)

        return {
            "overall_risk": overall_risk,
            "clauses": [result.to_dict() for result in clause_results],
        }

    def generate_report(self, review_result: Dict[str, object]) -> str:
        """Format the structured review result as a readable report."""

        lines = [
            "AI 기반 계약서 분석 보고서",
            "============================",
            f"전체 위험도: {review_result['overall_risk'].upper()}",
            "",
        ]

        clauses = review_result.get("clauses", [])
        for clause in clauses:
            lines.extend(
                [
                    f"[{clause['name']} - 위험도: {clause['risk_level']}]",
                    clause.get("summary", ""),
                ]
            )
            matches = clause.get("matched_sentences") or []
            if matches:
                lines.append("  감지된 문장:")
                for sentence in matches:
                    lines.append(f"    - {sentence}")
            issues = clause.get("issues") or []
            if issues:
                lines.append("  이슈:")
                for issue in issues:
                    lines.append(f"    - {issue}")
            notes = clause.get("notes") or []
            if notes:
                lines.append("  참고:")
                for note in notes:
                    lines.append(f"    - {note}")
            recommendation = clause.get("recommendation")
            if recommendation:
                lines.append(f"  권장 조치: {recommendation}")
            lines.append("")

        if not clauses:
            lines.append("분석된 조항이 없습니다.")

        return "\n".join(lines)

    @staticmethod
    def _normalise_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in raw_sentences if sentence.strip()]

    @staticmethod
    def _build_keyword_pattern(keyword: str) -> Pattern[str]:
        """Create a regex pattern that matches keyword boundaries safely."""

        keyword = keyword.strip().lower()
        if not keyword:
            return re.compile(r"^$")

        if re.search(r"\s", keyword):
            parts = [re.escape(part) for part in keyword.split() if part]
            pattern = r"\b" + r"\s+".join(parts) + r"\b"
        elif re.search(r"\w", keyword):
            pattern = r"\b" + re.escape(keyword) + r"\b"
        else:
            pattern = re.escape(keyword)

        return re.compile(pattern)

    def _evaluate_clause(self, config: ClauseConfig, sentences: Sequence[str]) -> ClauseResult:
        keyword_patterns = [self._build_keyword_pattern(keyword) for keyword in config.keywords]
        matched = [
            sentence
            for sentence in sentences
            if any(pattern.search(sentence) for pattern in keyword_patterns)
        ]
        issues: List[str] = []
        notes: List[str] = []
        risk = "low" if matched else config.missing_risk

        if matched:
            warnings, positives = self._evaluate_warnings(config, matched)
            issues.extend(warnings)
            notes.extend(positives)
            if warnings and risk != "high":
                risk = "medium" if risk == "low" else risk
                if any('심각' in warning for warning in warnings):
                    risk = "high"

        recommendation = config.recommendation

        if not matched:
            issues = [f"{config.name} 조항이 감지되지 않았습니다."]

        return ClauseResult(
            name=config.name,
            present=bool(matched),
            risk_level=risk,
            matched_sentences=matched,
            issues=issues,
            notes=notes,
            recommendation=recommendation,
            summary=config.summary,
        )

    def _evaluate_warnings(self, config: ClauseConfig, sentences: Sequence[str]) -> Tuple[List[str], List[str]]:
        warnings: List[str] = []
        positives: List[str] = []
        for sentence in sentences:
            for warning_keyword in config.warning_keywords:
                if warning_keyword in sentence:
                    warnings.append(
                        f"주의: '{warning_keyword}' 표현이 포함되어 있어 위험이 증가할 수 있습니다."
                    )
            positive_hits = [kw for kw in config.positive_keywords if kw in sentence]
            if positive_hits:
                positives.append(
                    "양호: " + ", ".join(positive_hits) + " 표현이 있어 조건이 개선됩니다."
                )
        return warnings, positives

    @staticmethod
    def _calculate_overall_risk(results: Sequence[ClauseResult]) -> str:
        risk_levels = {"low": 0, "medium": 1, "high": 2}
        max_score = max((risk_levels.get(result.risk_level, 0) for result in results), default=0)
        for level, score in risk_levels.items():
            if score == max_score:
                return level
        return "low"


def _main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="AI 기반 계약서 검토 서비스")
    parser.add_argument("path", type=Path, help="검토할 계약서 파일 경로")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="출력 형식 선택 (기본: text)",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"계약서 파일을 찾을 수 없습니다: {args.path}")

    contract_text = args.path.read_text(encoding="utf-8")
    service = ContractReviewService()
    review = service.review(contract_text)

    if args.format == "json":
        print(json.dumps(review, ensure_ascii=False, indent=2))
    else:
        print(service.generate_report(review))


if __name__ == "__main__":
    _main()
