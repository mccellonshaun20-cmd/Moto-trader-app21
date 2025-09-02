class AdaptiveWeighter:
    def __init__(self, alpha=0.2):
        self.w = {'tech': 1/3, 'macro': 1/3, 'fund': 1/3}
        self.alpha = alpha
        self.hist = {'tech': [], 'macro': [], 'fund': []}
    def update(self, pnl: float, contrib: dict):
        for k in self.w:
            c = contrib.get(k, 0.0)
            hit = 1.0 if pnl*c >= 0 else -1.0
            self.hist[k].append(hit)
            perf = sum(self.hist[k][-100:]) / max(1, len(self.hist[k][-100:]))
            self.w[k] = (1-self.alpha)*self.w[k] + self.alpha*max(0.0, perf+0.5)
        s = sum(self.w.values())
        for k in self.w: self.w[k] /= s
