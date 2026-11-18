from pathlib import Path

from retrieval import normalize


class E5Retriever:
    """CPU retrieval with a local multilingual E5-small ONNX export."""

    def __init__(self, model_dir):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(model_dir / "onnx" / "model_int8.onnx"), options, providers=["CPUExecutionProvider"],
        )
        self.inputs = {item.name for item in self.model.get_inputs()}
        self.index = {}

    def encode(self, texts):
        batch = self.tokenizer.encode_batch(texts)
        fields = {
            "input_ids": [item.ids for item in batch],
            "attention_mask": [item.attention_mask for item in batch],
            "token_type_ids": [item.type_ids for item in batch],
        }
        mask = self.np.asarray(fields["attention_mask"], dtype=self.np.int64)[..., None]
        hidden, = self.model.run(None, {
            name: self.np.asarray(fields[name], dtype=self.np.int64) for name in self.inputs
        })
        vectors = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        return vectors / self.np.linalg.norm(vectors, axis=1, keepdims=True)

    def fit(self, corpus):
        embeddings = self.encode([f"passage: {normalize(row.search_text)}" for row in corpus])
        self.index = {row.id: embedding for row, embedding in zip(corpus, embeddings, strict=True)}

    def scores(self, question, candidates):
        query, = self.encode([f"query: {question}"])
        return {row.id: float(self.index[row.id] @ query) for row in candidates}
