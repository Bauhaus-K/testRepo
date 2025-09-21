import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_review_service import ContractReviewService


def test_detects_missing_termination_clause():
    service = ContractReviewService()
    contract_text = (
        "Confidential information shared under this agreement remains proprietary. "
        "Fees are payable within thirty days of invoice."
    )

    review = service.review(contract_text)
    clauses = {clause["name"]: clause for clause in review["clauses"]}

    termination = clauses["Termination"]
    assert termination["present"] is False
    assert termination["risk_level"] == "high"
    assert any("감지되지 않았습니다" in issue for issue in termination["issues"])
    assert review["overall_risk"] == "high"


def test_positive_notes_lower_risk_when_clauses_are_balanced():
    service = ContractReviewService()
    contract_text = (
        "Either party may terminate this agreement upon thirty days written notice. "
        "The parties agree to limit liability with a maximum cap on damages. "
        "Confidential information must be returned or destroyed within ten days."
    )

    review = service.review(contract_text)
    clauses = {clause["name"]: clause for clause in review["clauses"]}

    termination = clauses["Termination"]
    assert termination["present"] is True
    assert termination["risk_level"] == "low"
    assert any(note.startswith("양호") for note in termination["notes"])

    liability = clauses["Liability"]
    assert liability["risk_level"] == "low"
    assert any(note.startswith("양호") for note in liability["notes"])

    assert review["overall_risk"] == "medium"
