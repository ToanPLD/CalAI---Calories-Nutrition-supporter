import hashlib


class DedupService:

    def __init__(self):
        self.seen = set()

    def is_duplicate(self, payload):
        h = hashlib.md5(str(payload).encode()).hexdigest()

        if h in self.seen:
            return True

        self.seen.add(h)
        return False