# hybrid_model.py

class HybridRecommendationSystem_Custom:
    def __init__(self, cf_model=None, cb_model=None, weights=(0.5, 0.5)):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.weights = weights

    def fit(self, X):
        if self.cf_model:
            self.cf_model.fit(X)
        if self.cb_model:
            self.cb_model.fit(X)

    def predict(self, user_id, item_id):
        score_cf, score_cb = 0, 0
        if self.cf_model:
            score_cf = self.cf_model.predict(user_id, item_id).est
        if self.cb_model:
            score_cb = self.cb_model.predict(user_id, item_id)
        return self.weights[0] * score_cf + self.weights[1] * score_cb

    def recommend(self, user_id, items, top_n=10):
        scored_items = []
        for item_id in items:
            try:
                score = self.predict(user_id, item_id)
                scored_items.append((item_id, score))
            except Exception:
                continue
        return sorted(scored_items, key=lambda x: x[1], reverse=True)[:top_n]
