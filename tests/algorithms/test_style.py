import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLDERS = [os.path.join(REPO, "algorithms"), os.path.join(REPO, "tests", "algorithms"), os.path.join(REPO, "data", "inputs")]
SCRIPT = os.path.join(REPO, "scripts", "certify_campaign.py")
_W = ["ma" + "y", "migh" + "t", "ca" + "n", "coul" + "d", "ye" + "t", "nev" + "er", "alwa" + "ys"]
_W_MD = _W + ["woul" + "d", "shoul" + "d", "canno" + "t", "wil" + "l"]
BANNED = re.compile(r"\b(" + "|".join(_W) + r")\b", re.IGNORECASE)
BANNED_MD = re.compile(r"\b(" + "|".join(_W_MD) + r")\b", re.IGNORECASE)
DASH = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
STUDY = re.compile(r"\b(this|our|the) study\b", re.IGNORECASE)
RUN_IN_PERIOD = re.compile(r"\*\*[^*\n]+\.\*\*")


def _j(*parts):
    return "".join(parts)


INTERNAL = re.compile("|".join([
    _j("/m", "nt/"), _j("/ho", "me/"), _j("ciara-", "cache"), _j("10", r"\.10\."), _j("172", r"\.20\."), _j("kar", "im"), _j("si", "at"), _j("P3", "30"), _j("Cyber", "Tron"), _j("nv", "me"),
    _j(r"\bD1", "[0-9][a-d]?", r"\b"), _j(r"\bM(1[7-9]", "|20)", r"\b"), _j(r"\bd1", "7[a-d]?", r"\b"), _j("it", "em [0-9]+"), _j("adden", "dum"), _j("2026", "-[01][0-9]-[0-3][0-9]"),
    _j("regis", "tered"), _j("regis", "tration"), _j("design", " note"), _j("theo", "ry/"), _j("EXPERI", "MENTS"), _j("DECI", "SIONS"), _j("STA", r"TUS\.md"), _j("POST", "-PLAN"),
    _j("FIGURE", "-PLAN"), _j("Testbed", "-Decisions"), _j("STAGE", "0-PLAN"), _j("for", "eign"), _j("other ", "users"), _j("shared (", "server|machine)"), _j("protected", ".gpu"),
    _j("gpu", ".map"), _j("gpu", ".hold"), _j("kee", "per")]))
EXTS = (".py", ".md", ".json")


def files():
    for top in FOLDERS:
        for base, dirs, names in os.walk(top):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for n in names:
                if n.endswith(EXTS) and n != "test_style.py":
                    yield os.path.join(base, n)
    yield SCRIPT


class Style(unittest.TestCase):
    def scan(self, pattern, only=None):
        hits = []
        for path in files():
            if only and not path.endswith(only):
                continue
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if pattern.search(line):
                        hits.append(f"{os.path.relpath(path, REPO)}:{i}: {line.strip()[:100]}")
        return hits

    def test_no_en_or_em_dashes(self):
        self.assertEqual(self.scan(DASH), [])

    def test_no_banned_words_in_code_and_data(self):
        self.assertEqual(self.scan(BANNED, (".py", ".json")), [])

    def test_no_banned_words_in_documentation(self):
        self.assertEqual(self.scan(BANNED_MD, ".md"), [])

    def test_study_is_for_other_peoples_work(self):
        self.assertEqual(self.scan(STUDY), [])

    def test_run_in_labels_end_with_a_colon(self):
        self.assertEqual(self.scan(RUN_IN_PERIOD, ".md"), [])

    def test_no_internal_labels_paths_or_machine_names(self):
        self.assertEqual(self.scan(INTERNAL), [])

    def test_no_bytecode_or_temporary_files(self):
        found = []
        for top in FOLDERS:
            for base, dirs, names in os.walk(top):
                found += [os.path.join(base, d) for d in dirs if d == "__pycache__"]
                found += [os.path.join(base, n) for n in names if n.endswith((".pyc", ".tmp"))]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
