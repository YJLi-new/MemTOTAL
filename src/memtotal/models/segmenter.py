from __future__ import annotations


class Segmenter:
    def __init__(self, mode: str = "delimiter", delimiter: str = "||") -> None:
        if mode != "delimiter":
            raise NotImplementedError("M0 bootstrap currently supports only delimiter segmentation.")
        self.mode = mode
        self.delimiter = delimiter

    def split(self, text: str) -> list[str]:
        segments = [segment.strip() for segment in text.split(self.delimiter)]
        cleaned = [segment for segment in segments if segment]
        return cleaned or [text.strip()]

