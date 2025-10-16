"""A matching view that removes controls without losing original evidence spans."""

from bisect import bisect_left
from dataclasses import dataclass
import unicodedata


def is_removable_control(char: str) -> bool:
    category = unicodedata.category(char)
    return category == "Cf" or (category == "Cc" and char not in "\t\n\r")


def without_format_controls(text: str) -> str:
    return "".join(char for char in text if not is_removable_control(char))


@dataclass(frozen=True)
class SourceTextView:
    original: str
    text: str
    offsets: tuple[int, ...]

    @classmethod
    def from_text(cls, text: str):
        offsets = tuple(index for index, char in enumerate(text) if not is_removable_control(char))
        return cls(text, "".join(text[index] for index in offsets), offsets)

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.offsets):
            raise ValueError("Invalid matching-view span")
        return self.offsets[start], self.offsets[end - 1] + 1

    def matching_span(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.original):
            raise ValueError("Invalid source span")
        return bisect_left(self.offsets, start), bisect_left(self.offsets, end)

    def restore_evidence(self, item: dict) -> dict:
        start, end = self.original_span(*item["position"])
        return {**item, "position": (start, end), "raw": self.original[start:end]}
