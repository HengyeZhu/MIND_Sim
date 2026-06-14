from __future__ import annotations

import subprocess


def _nmodl_random_syntax_fixture(
    *,
    s0: str = ":s0",
    s1: str = "RANDOM rv1, rv2",
    s2: str = ":s2",
    s3: str = ":s3",
    s4: str = "x1 = random_negexp(rv1, 1.0)",
    s5: str = ":s5",
    s6: str = "foo = random_negexp(rv1, 1.0)",
) -> str:
    return f"""
: temp.mod file with format elements to test for RANDOM syntax errors

{s0} : 0 error if randomvar is mentioned

NEURON {{
  SUFFIX temp
  RANGE x1
  {s1} : 1 declare randomvars
}}

{s2} : 2 error if randomvar is mentioned

ASSIGNED {{
  x1
}}

BEFORE STEP {{
  {s3} : 3 error if assign or eval a randomvar
  {s4} : 4 random_function accepted but ranvar must be first arg
}}

FUNCTION foo(arg) {{
  {s5} : 5  LOCAL ranvar makes it a double in this scope
  {s6} : 6  random_function accepted but ranvar must be first arg
  foo = arg
}}
"""


def test_syntax(tmp_path):
    def run_case(name, text, expect_success):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "temp.mod").write_text(text)
        result = subprocess.run(
            ["mind_nrnivmodl", str(directory)],
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) is expect_success, result.stderr + result.stdout

    cases = [
        ("valid", _nmodl_random_syntax_fixture(), True),
        ("s0_assigned_random", _nmodl_random_syntax_fixture(s0="ASSIGNED{rv1}"), False),
        ("s0_local_random", _nmodl_random_syntax_fixture(s0="LOCAL rv1"), False),
        ("s2_assigned_random", _nmodl_random_syntax_fixture(s2="ASSIGNED{rv1}"), False),
        ("s2_local_random", _nmodl_random_syntax_fixture(s2="LOCAL rv1"), False),
        ("s3_assign_random", _nmodl_random_syntax_fixture(s3="rv1 = 1"), False),
        ("s3_eval_random", _nmodl_random_syntax_fixture(s3="x1 = rv1"), False),
        ("s4_random_as_arg", _nmodl_random_syntax_fixture(s4="foo(rv1)"), False),
        ("s4_no_args", _nmodl_random_syntax_fixture(s4="random_negexp()"), False),
        ("s4_missing_random_first_arg", _nmodl_random_syntax_fixture(s4="random_negexp(1.0)"), False),
        ("s5_local_random", _nmodl_random_syntax_fixture(s5="LOCAL rv1"), False),
        ("s4_random_second_arg", _nmodl_random_syntax_fixture(s4="random_negexp(rv1, rv2)"), False),
    ]

    for name, text, expect_success in cases:
        run_case(name, text, expect_success)
