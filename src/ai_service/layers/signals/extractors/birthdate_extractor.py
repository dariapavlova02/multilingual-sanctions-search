"""
Birthdate extractor.
"""

from typing import Any, Dict, List

from .base_extractor import BaseExtractor
from ....utils.source_text_view import SourceTextView


class BirthdateExtractor(BaseExtractor):
    """Extracts birthdates from text."""

    def extract(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract birthdates from text.

        Args:
            text: Input text

        Returns:
            List of birthdate dictionaries with date, raw, and position info
        """
        if not self._is_valid_text(text):
            return []

        try:
            from ....data.patterns.dates import extract_birthdates_from_text

            view = SourceTextView.from_text(text)
            birthdates = [view.restore_evidence(item) for item in extract_birthdates_from_text(view.text)]
            self._log_extraction_result(text, len(birthdates), "birthdate")
            return birthdates

        except ImportError:
            self.logger.warning(
                "Date patterns not available, falling back to empty result"
            )
            return []
