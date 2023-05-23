import numpy as np
from sentence_transformers.util import cos_sim
from configs.config import THRESHOLD


class PlagiarismModel:
    instance = None
    
    def __new__(cls, encoder_model):
        if not cls.instance:
            cls.instance = super(PlagiarismModel, cls).__new__(cls)
            try:
                cls.model = encoder_model.model
            except AttributeError as err:
                cls.model = encoder_model
        return cls.instance

    def __embed_corpus(self, corpus, batch=50):
        if not isinstance(corpus, list):
            corpus = [corpus]
        corpus = list(map(str, corpus))
        corpus_ls = []
        n_corpus = len(corpus)
        for i in range(0, n_corpus, batch):
            corpus_emb = self.model.encode(corpus[i:i+batch])
            corpus_ls.extend(corpus_emb)
        if n_corpus % batch != 0 and n_corpus > batch:
            corpus_emb = self.model.encode(corpus[i+batch:])
            corpus_ls.extend(corpus_emb)
        return np.array(corpus_ls)
    
    def __siamese_model(self, students_emb):
        sims = np.array(list(map(
            lambda s_emb: np.array(
                        cos_sim(s_emb.reshape(1, -1), students_emb)), students_emb)))
        sims = np.array(list(map(
            lambda sim: np.delete(sim[1], sim[0], axis=1), enumerate(sims.tolist())
        )))
        sims = np.array(list(map(
            lambda sim: np.insert(sim[1], sim[0], -np.inf, axis=1), enumerate(sims.tolist())
        )))
        return sims

    def __pligarism_pipeline(self, students_answers, ids, threshold=THRESHOLD):
        students_emb = self.__embed_corpus(students_answers)
        sims = self.__siamese_model(students_emb)
        res = list(map(lambda sim:
                       dict(zip(list(
                                map(lambda x: ids[x], 
                                    np.where(sim >= threshold)[1].tolist())), 
                                sim[sim >= threshold]
                                )), sims))
        return res
    
    def dummy_predict(self, answers):
        return [{1: 0.8, 2: 0.5, 3: 0.2},
                {4: 0.8, 5: 0.5}, answers]

    def predict(self, students_answers, ids, threshold=THRESHOLD):
        scores = self.__pligarism_pipeline(students_answers, ids, threshold)
        res = list(map(lambda r:
                       {r[0]: r[1]},
                       filter(lambda r:
                              any(r[1]),
                              zip(ids, scores)
                              )))
        return res