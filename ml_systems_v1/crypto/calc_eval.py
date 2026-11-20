"""calc_eval.py - safe arithmetic evaluator (no eval/exec/compile).

Tokenizer + recursive-descent parser over + - * / and parentheses for ints
and floats. Raises ValueError on malformed input and on division by zero.

Grammar:
    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := ('+' | '-') factor | NUMBER | '(' expr ')'
"""

from __future__ import annotations


def tokenize(text):
    """Convert input string into ('NUM', value) / ('OP', char) tokens."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c in "+-*/()":
            tokens.append(("OP", c))
            i += 1
            continue
        if c.isdigit() or c == ".":
            start = i
            seen_dot = False
            while i < n and (text[i].isdigit() or text[i] == "."):
                if text[i] == ".":
                    if seen_dot:
                        raise ValueError(f"malformed number at position {start}")
                    seen_dot = True
                i += 1
            num_str = text[start:i]
            if num_str == ".":
                raise ValueError(f"malformed number at position {start}")
            value = float(num_str) if seen_dot else int(num_str)
            tokens.append(("NUM", value))
            continue
        raise ValueError(f"unexpected character {c!r} at position {i}")
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def parse(self):
        if not self.tokens:
            raise ValueError("empty expression")
        result = self.expr()
        if self.pos != len(self.tokens):
            raise ValueError("unexpected trailing tokens")
        return result

    def expr(self):
        value = self.term()
        while True:
            tok = self._peek()
            if tok == ("OP", "+"):
                self._advance()
                value = value + self.term()
            elif tok == ("OP", "-"):
                self._advance()
                value = value - self.term()
            else:
                break
        return value

    def term(self):
        value = self.factor()
        while True:
            tok = self._peek()
            if tok == ("OP", "*"):
                self._advance()
                value = value * self.factor()
            elif tok == ("OP", "/"):
                self._advance()
                divisor = self.factor()
                if divisor == 0:
                    raise ValueError("division by zero")
                value = value / divisor
            else:
                break
        return value

    def factor(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of input")
        if tok == ("OP", "+"):
            self._advance()
            return self.factor()
        if tok == ("OP", "-"):
            self._advance()
            return -self.factor()
        if tok == ("OP", "("):
            self._advance()
            value = self.expr()
            closing = self._advance()
            if closing != ("OP", ")"):
                raise ValueError("expected closing parenthesis")
            return value
        if tok[0] == "NUM":
            self._advance()
            return tok[1]
        raise ValueError(f"unexpected token {tok!r}")


def calc_eval(text):
    """Safely evaluate an arithmetic expression string."""
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    tokens = tokenize(text)
    return _Parser(tokens).parse()


if __name__ == "__main__":
    assert calc_eval("2+3*4") == 14, calc_eval("2+3*4")
    assert calc_eval("(2+3)*4") == 20, calc_eval("(2+3)*4")
    assert calc_eval("8/2") == 4.0, calc_eval("8/2")
    assert calc_eval("7/2") == 3.5, calc_eval("7/2")
    for bad in ("1+", "1/0"):
        try:
            calc_eval(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    print("all self-checks passed")
