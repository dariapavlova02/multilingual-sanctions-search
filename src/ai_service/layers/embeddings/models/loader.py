"""One pinned, safetensors-only loader for every embedding entry point."""

from ....config import EmbeddingConfig


def load_embedding_model(config: EmbeddingConfig, *, cache_folder=None,
                         use_fp16=False, expected_max_sequence_length=None):
    from sentence_transformers import SentenceTransformer

    if use_fp16 and config.device == "cpu":
        raise ValueError("FP16 requires an explicitly selected accelerator")
    kwargs = {"device": config.device, "revision": config.revision,
              "trust_remote_code": False, "model_kwargs": {"use_safetensors": True}}
    if cache_folder is not None:
        kwargs["cache_folder"] = cache_folder
    model = SentenceTransformer(config.model_name, **kwargs)
    if model.get_sentence_embedding_dimension() != config.dimension:
        raise ValueError("Model dimension differs from the configured index contract")
    # Truncation is fixed by the pinned artifact. A legacy setting may verify it,
    # but cannot silently change the vector space without a different contract.
    if expected_max_sequence_length is not None and model.max_seq_length != expected_max_sequence_length:
        raise ValueError("Sequence length differs from the pinned model contract")
    if use_fp16:
        model = model.half()
    return model
