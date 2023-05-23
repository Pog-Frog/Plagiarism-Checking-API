from typing import Optional
from sentence_transformers import SentenceTransformer
from configs.config import MODELPATH


class BERTModel(object):
    def __new__(
        cls: object, modelpath: Optional[str]=MODELPATH):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BERTModel, cls).__new__(cls)
            cls.model = SentenceTransformer(modelpath)
        return cls.instance
