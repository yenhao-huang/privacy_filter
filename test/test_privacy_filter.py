from pathlib import Path
import tempfile
import unittest

from core.service.privacy_filter import PrivacyFilter, load_pattern_config


class PrivacyFilterTest(unittest.TestCase):
    def test_redacts_default_sensitive_values(self) -> None:
        filterer = PrivacyFilter()

        redacted = filterer.redact(
            "Email jane.doe@example.com, call 555-123-4567, ssn 123-45-6789."
        )

        self.assertEqual(
            redacted,
            "Email [REDACTED:email], call [REDACTED:phone], ssn [REDACTED:ssn].",
        )

    def test_lists_matches_in_text_order(self) -> None:
        filterer = PrivacyFilter()

        matches = filterer.find("token=abcdef1234567890 user@example.com")

        self.assertEqual([match.kind for match in matches], ["api_key", "email"])

    def test_loads_custom_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "patterns.json"
            path.write_text(
                '{"replacement":"<{kind}>","patterns":[{"kind":"account","pattern":"ACCT-[0-9]{4}"}]}',
                encoding="utf-8",
            )

            config = load_pattern_config(path)
            filterer = PrivacyFilter(patterns=config.patterns, replacement=config.replacement)

        self.assertEqual(filterer.redact("Use ACCT-1234"), "Use <account>")


if __name__ == "__main__":
    unittest.main()

