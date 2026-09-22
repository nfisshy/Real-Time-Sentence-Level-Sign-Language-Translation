from __future__ import annotations


def _common_prefix_tokens(left: list[str], right: list[str]) -> list[str]:
    prefix: list[str] = []
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix.append(left_token)
    return prefix


class TextStabilizer:
    """
    Keeps a fast draft output and a conservative stable output.
    """

    def __init__(self):
        self._previous_tokens: list[str] = []
        self._stable_tokens: list[str] = []

    def reset(self) -> None:
        self._previous_tokens = []
        self._stable_tokens = []

    def update(self, hypothesis: str) -> dict[str, object]:
        tokens = hypothesis.split()
        changed = False

        if self._previous_tokens:
            common_prefix = _common_prefix_tokens(self._previous_tokens, tokens)
            if len(common_prefix) > len(self._stable_tokens):
                self._stable_tokens = common_prefix
                changed = True
            if tokens == self._previous_tokens and tokens != self._stable_tokens:
                self._stable_tokens = list(tokens)
                changed = True
        elif tokens:
            changed = True

        self._previous_tokens = list(tokens)
        return {
            "draft_text": " ".join(tokens),
            "stable_text": " ".join(self._stable_tokens),
            "stable_changed": changed,
        }
