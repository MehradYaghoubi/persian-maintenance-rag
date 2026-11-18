import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("CN-120", "CN-240")
REVISIONS = ("1.0", "2.0")
CODE = re.compile(r"(?<![A-Z0-9])E[\s-]*(\d{1,5})(?![A-Z0-9])")
MODEL = re.compile(r"\bCN[\s-]*(\d{2,4})\b", re.ASCII)
REVISION = re.compile(r"(?:نسخه|ویرایش|ورژن|\bV|\bREV(?:ISION)?)\s*[:\-]?\s*(\d+\.\d+)", re.ASCII)
NORMALIZE = str.maketrans("يك۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩‌٫", "یک01234567890123456789 .")
UNDOCUMENTED_ACTION = re.compile(
    r"\b(?:غیر\s*فعال|دور\s*زدن|دور\s*بزن\w*|حذف|پاک\s+کن\w*|"
    r"سیم\s*کشی|تعویض|خرید|بخر\w*|ریست|راه\s*اندازی\s*مجدد|آدرس\s+IP)\b"
    r"|\b(?:چند\s+(?:درجه|ولت|آمپر|درصد)|(?:چقدر|چه\s+درصد)\s+[^؟?]{0,24}مجاز|"
    r"(?:تنظیم|ست)\s+(?:روی\s+)?\d+|روی\s+\d+\s+(?:تنظیم|ست))\b"
)
STOP = set(
    "چه چیست چیه کجا چی چرا چگونه چطور را در از تا برای با به و یا که این آن یک هم "
    "کند کنم کنید می شود شده هست است دارد دارم دارد بدهید ثبت گزارش پیدا نشان کد خطا "
    "کنترلر دستگاه مدل نسخه ویرایش ورژن لطفا یعنی بشه بوده باید بدون آیا بر دارد تو "
    "کن بگو بکنیم معنی معنای بهم کدام مقدار عدد صفحه دید دیدن دربیارم نشون باز مربوط".split()
)
ALIASES = {
    "ورژن فریمور": "نسخه نرم افزار",
    "فریمور": "نرم افزار",
    "میان افزار": "نرم افزار",
    "بیلد": "نرم افزار",
    "قدیمی ها": "تاریخچه",
    "آلارم های قبلی": "سابقه تاریخچه",
    "هشدارای قبلی": "سابقه تاریخچه",
    "لیست آلارم": "سابقه دفتر هشدارها",
    "رخدادهای پیشین": "سابقه تاریخچه",
    "سوابق": "سابقه تاریخچه",
    "برق ورودی": "ولتاژ ورودی تغذیه",
    "برق تغذیه": "ولتاژ ورودی تغذیه",
    "ولت": "ولتاژ ورودی تغذیه",
    "نیروی کشش": "کشش",
    "داغی قاب": "دمای محفظه",
    "فوتوسل": "حسگر نوری",
    "دو شاخه": "دوشاخه",
    "شماره ورژن": "نسخه نرم افزار",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(NORMALIZE)
    return " ".join("".join(c for c in text if unicodedata.category(c) != "Mn").upper().split())


def tokens(text: str) -> set[str]:
    return {term for term in re.findall(r"[^\W\d_]+", text) if len(term) > 1} - STOP


def normalized_terms(text: str) -> set[str]:
    # shortcut: handle productive Persian plurals, add a morphological analyzer if the manuals grow.
    return {
        re.sub(r"(?:هایی|های|ها)$", "", word) if len(word) > 4 else word
        for word in tokens(text)
    }


ALIASES = {normalize(alias): normalize(terms) for alias, terms in ALIASES.items()}
ALIAS_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, ALIASES)) + r")\b")


@dataclass(frozen=True)
class Evidence:
    id: str
    title: str
    model: str
    revision: str
    section: str
    code: str | None
    excerpt: str

    @property
    def search_text(self) -> str:
        return f"{self.code or ''} {self.section} {self.excerpt}".strip()


def load_manuals() -> list[Evidence]:
    corpus = []
    for model in MODELS:
        for revision in REVISIONS:
            title, *sections = (ROOT / "manuals" / f"{model}-{revision}.md").read_text(encoding="utf-8").split("\n## ")
            for section in sections:
                heading, excerpt = section.split("\n", 1)
                key, label = heading.split(" | ", 1)
                corpus.append(Evidence(
                    f"{model}/{revision}/{key}", title.removeprefix("# ").strip(), model,
                    revision, label, key if CODE.fullmatch(key) else None, excerpt.strip(),
                ))
    return corpus


def outcome(status: str, answer: str, evidence: tuple | list = (), missing: tuple | list = ()) -> dict:
    return {
        "status": status, "answer": answer, "citations": [asdict(row) for row in evidence],
        "missing": list(missing), "human_review": status == "abstain",
    }


def clarify(fields: list[str]) -> dict:
    labels = {"model": "مدل کنترلر (CN-120 یا CN-240)", "revision": "نسخه راهنما (1.0 یا 2.0)"}
    return outcome("clarify", f"لطفاً {' و '.join(labels[field] for field in fields)} را مشخص کنید.", missing=fields)


def abstain() -> dict:
    return outcome("abstain", "در راهنماهای موجود شاهد کافی پیدا نشد. این پرسش به بررسی مسئول نگهداری نیاز دارد.")


class Engine:
    def __init__(self, semantic=None, *, dense=None):
        self.corpus = load_manuals()
        self.semantic = semantic
        self.dense = dense
        self.retrieval = semantic.retrieval if semantic else "hybrid-e5" if dense else "hybrid-rules"
        if dense is not None:
            dense.fit(self.corpus)
        self.terms = {row.id: tokens(normalize(row.search_text)) for row in self.corpus}
        self.normalized = {row.id: normalized_terms(normalize(row.search_text)) for row in self.corpus}
        self.headings = {row.id: normalized_terms(normalize(row.section)) for row in self.corpus}

    def ask(self, question: str, model=None, revision=None, *, mode="hybrid") -> tuple[dict, list[Evidence]]:
        query = normalize(question)
        # The shipped manuals contain status and observation instructions only.
        if UNDOCUMENTED_ACTION.search(query):
            return abstain(), []
        models = {f"CN-{value}" for value in MODEL.findall(query)}
        revisions = set(REVISION.findall(query))
        conflicts = [
            field for field, values, explicit in (
                ("model", models, model), ("revision", revisions, revision),
            ) if len(values) > 1 or (values and explicit is not None and explicit not in values)
        ]
        if conflicts:
            return clarify(conflicts), []
        model = next(iter(models), model)
        revision = next(iter(revisions), revision)
        scope = [(m, r) for m in MODELS for r in REVISIONS if model in (None, m) and revision in (None, r)]
        candidates = [row for row in self.corpus if (row.model, row.revision) in scope]
        codes = {f"E{value}" for value in CODE.findall(query)}
        if not candidates or len(codes) > 1:
            return abstain(), []
        if codes:
            matches = [row for row in candidates if row.code in codes]
            return self.resolve(matches, scope), matches

        query = REVISION.sub(" ", MODEL.sub(" ", query)).strip()
        words = tokens(query)
        if not words:
            return abstain(), []

        literal = {row.id: len(self.terms[row.id] & words) / len(words) for row in candidates}
        scores = literal
        concept_words = set()
        if mode == "hybrid" and self.semantic is None:
            expanded = ALIAS_PATTERN.sub(lambda match: ALIASES[match[0]], query)
            concept_words = normalized_terms(expanded)
            if concept_words:
                scores = {
                    row.id: max(literal[row.id], len(self.normalized[row.id] & concept_words) / len(concept_words))
                    for row in candidates
                }
        if mode == "hybrid" and self.semantic is not None:
            vector = self.semantic.similar(query, model, revision)
            scores = {
                row.id: max(literal[row.id], vector.get(row.id, 0.0)) for row in candidates
            }

        ranked = sorted((row for row in candidates if scores[row.id] > 0), key=lambda row: (-scores[row.id], row.id))
        shortlist = [row for row in ranked if scores[row.id] >= 0.55]
        if not shortlist and concept_words and ranked:
            top = ranked[0]
            key = top.id.rsplit("/", 1)[1]
            competing_score = next((scores[row.id] for row in ranked if row.id.rsplit("/", 1)[1] != key), 0.0)
            if (
                scores[top.id] >= 0.42
                and scores[top.id] - competing_score >= 0.16
                and len(self.normalized[top.id] & concept_words) >= 2
                and self.headings[top.id] & concept_words
            ):
                shortlist = [top]
        if mode == "hybrid" and self.dense is not None:
            dense_scores = self.dense.scores(query, candidates)
            dense_ranked = sorted(candidates, key=lambda row: (-dense_scores[row.id], row.id))
            if not shortlist:
                top = dense_ranked[0]
                key = top.id.rsplit("/", 1)[1]
                runner_up = next((dense_scores[row.id] for row in dense_ranked if row.id.rsplit("/", 1)[1] != key), 0.0)
                if dense_scores[top.id] >= 0.82 and dense_scores[top.id] - runner_up >= 0.035:
                    shortlist = [top]
            # Interleave lexical and dense ranks so either signal can recover a citation.
            rank_indices = [{row.id: index for index, row in enumerate(order)} for order in (ranked, dense_ranked)]
            ranked = sorted(candidates, key=lambda row: (
                -sum(1 / (10 + ranks[row.id]) for ranks in rank_indices if row.id in ranks), row.id,
            ))
        if not shortlist:
            return abstain(), ranked

        selected = shortlist[0]
        if mode == "hybrid" and self.semantic is not None:
            selected = self.semantic.select(question, shortlist[:5])
            if selected is None:
                return abstain(), ranked

        key = selected.id.rsplit("/", 1)[1]
        matches = [row for row in candidates if row.id.rsplit("/", 1)[1] == key]
        return self.resolve(matches, scope), ranked

    @staticmethod
    def resolve(matches: list[Evidence], scope: list[tuple[str, str]]) -> dict:
        if not matches:
            return abstain()
        by_scope = {(row.model, row.revision): row.excerpt for row in matches}
        if len({by_scope.get(pair) for pair in scope}) > 1:
            missing = []
            if any(
                by_scope.get((m, r)) != by_scope.get((other, r))
                for m, r in scope for other, same_revision in scope if r == same_revision
            ):
                missing.append("model")
            if any(
                by_scope.get((m, r)) != by_scope.get((m, other))
                for m, r in scope for same_model, other in scope if m == same_model
            ):
                missing.append("revision")
            return clarify(missing)
        return outcome("answered", matches[0].excerpt, matches)
