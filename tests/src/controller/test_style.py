import os
import re
import unittest

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "src", "controller")
BANNED = re.compile(r"\b(may|might|can|could|yet|never|always)\b", re.IGNORECASE)
DASH = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
STUDY = re.compile(r"\b(this|our|the) study\b", re.IGNORECASE)
RUN_IN_PERIOD = re.compile(r"\*\*[^*\n]+\.\*\*")
EXTS = (".py", ".yaml", ".yml", ".json", ".sh", ".md", ".txt")


def files():
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for n in names:
            if n.endswith(EXTS) and n != "test_style.py":
                yield os.path.join(base, n)


class Style(unittest.TestCase):
    def scan(self, pattern):
        hits = []
        for path in files():
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if pattern.search(line):
                        hits.append(f"{os.path.relpath(path, ROOT)}:{i}: {line.strip()[:100]}")
        return hits

    def test_no_en_or_em_dashes(self):
        self.assertEqual(self.scan(DASH), [])

    def test_no_banned_words(self):
        self.assertEqual(self.scan(BANNED), [])

    def test_no_study_for_own_work(self):
        self.assertEqual(self.scan(STUDY), [])

    def test_run_in_labels_end_with_a_colon(self):
        self.assertEqual(self.scan(RUN_IN_PERIOD), [])


if __name__ == "__main__":
    unittest.main()
