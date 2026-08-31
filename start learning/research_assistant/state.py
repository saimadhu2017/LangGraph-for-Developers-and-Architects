import operator
from typing import Annotated, TypedDict


class Source(TypedDict):
    title: str
    text: str


class ResearchState(TypedDict):
    question: str                                    # what the user asked
    keywords: list[str]                              # OVERWRITTEN — the retry rewrites the query
    sources: Annotated[list[Source], operator.add]   # ACCUMULATES across searches
    summary: str                                     # the drafted answer
    verdict: str                                     # evaluator's judgement — now a ROUTING input
    attempts: Annotated[int, operator.add]           # NEW (v0.3) — how many times `search` has run
