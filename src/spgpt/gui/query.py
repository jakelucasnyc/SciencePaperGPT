from PySide6.QtCore import QRunnable, Signal, QObject
from PySide6.QtWidgets import QListWidgetItem
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from spgpt.query import get_response_from_query


class QuerySignals(QObject):
    finished = Signal()
    error = Signal(str)
    response_acquired = Signal(str)

class Query(QRunnable):

    def __init__(self, faiss_db:FAISS, query:str, temperature:float, k:int=8):
        super().__init__()
        self.signals = QuerySignals()
        self._faiss_db = faiss_db
        self._query = query
        self._temperature = temperature
        self._k = k

    def run(self):
        try:
            response, _ = get_response_from_query(self._faiss_db, self._query, self._temperature, self._k)
        except Exception as e:
            self.signals.finished.emit()
            self.signals.error.emit(repr(e))
        else:
            self.signals.finished.emit()
            self.signals.response_acquired.emit(response)
