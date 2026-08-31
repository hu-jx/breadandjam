from dataclasses import dataclass, asdict

@dataclass
class Upload:
    name: str
    bytes: bytes
    prob_ai: float
    verdict: str
    timestamp: str

    def to_dict(self, include_bytes: bool = False):
        data = asdict(self)
        if not include_bytes:
            data.pop("bytes")
        return data