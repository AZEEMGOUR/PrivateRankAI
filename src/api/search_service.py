import gc
import json
import time
from pathlib import Path

import torch
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from src.ingestion.pdf_loader import load_pdf_chunks


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "real_documents"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
)

CHUNKS_FILE = (
    INDEX_DIR
    / "chunks.json"
)

EMBEDDINGS_FILE = (
    INDEX_DIR
    / "embeddings.pt"
)


EMBEDDING_MODEL_NAME = (
    "Qwen/Qwen3-Embedding-0.6B"
)

RERANKER_MODEL_NAME = (
    "Qwen/Qwen3-Reranker-0.6B"
)


RERANKER_PROMPT = (
    "Rank enterprise passages by the exact user intent. "
    "Respect dates, policy versions, employee status, region, "
    "entity IDs, numeric thresholds, and business context. "
    "Temporal rules are strict: if the query asks for a rule BEFORE, "
    "PRIOR TO, OLD, PREVIOUS, HISTORICAL, RETIRED, or SUPERSEDED "
    "relative to a date, prefer the passage explicitly valid before "
    "that date. For CURRENT, ACTIVE, or LATEST queries, prefer the "
    "active rule."
)


class PrivateRankSearchService:

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.embedding_model = None
        self.reranker = None

        self.chunks = []
        self.document_embeddings = None

        self.ready = False

    def load(self):
        print(
            "\n======================================"
        )
        print(
            "Loading PrivateRankAI search service"
        )
        print(
            "======================================"
        )

        print(
            "Device:",
            self.device,
        )

        if torch.cuda.is_available():
            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

        print(
            "\nLoading embedding model..."
        )

        self.embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device=self.device,
                model_kwargs={
                    "torch_dtype":
                    torch.float16
                }
                if self.device == "cuda"
                else {},
            )
        )

        print(
            "\nLoading reranker..."
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            device=self.device,
            max_length=256,
            prompts={
                "query":
                RERANKER_PROMPT
            },
            default_prompt_name="query",
        )

        self.load_index()

        self.ready = True

        print(
            "\nPrivateRankAI API service ready."
        )

    def load_index(self):

        if (
            not CHUNKS_FILE.exists()
            or not EMBEDDINGS_FILE.exists()
        ):
            print(
                "No saved document index found."
            )

            self.chunks = []
            self.document_embeddings = None

            return

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            self.chunks = json.load(
                file
            )

        self.document_embeddings = (
            torch.load(
                EMBEDDINGS_FILE,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.document_embeddings = (
            self.document_embeddings.to(
                self.device
            )
        )

        print(
            "Indexed chunks:",
            len(self.chunks),
        )

        print(
            "Embedding shape:",
            tuple(
                self.document_embeddings.shape
            ),
        )

    def build_index(self):

        if self.embedding_model is None:
            raise RuntimeError(
                "Embedding model is not loaded."
            )

        INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_files = sorted(
            DOCUMENTS_DIR.glob(
                "*.pdf"
            )
        )

        if not pdf_files:
            self.clear_index()

            return {
                "documents": 0,
                "chunks": 0,
                "embedding_shape": [0, 0],
            }

        all_chunks = []

        for pdf_file in pdf_files:

            chunks = load_pdf_chunks(
                pdf_file
            )

            print(
                f"Loaded {pdf_file.name}: "
                f"{len(chunks)} chunks"
            )

            all_chunks.extend(
                chunks
            )

        if not all_chunks:
            raise RuntimeError(
                "No chunks were created."
            )

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        print(
            "Creating document embeddings..."
        )

        embeddings = (
            self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
        )

        embeddings_cpu = (
            embeddings.detach().cpu()
        )

        with open(
            CHUNKS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                all_chunks,
                file,
                indent=2,
                ensure_ascii=False,
            )

        torch.save(
            embeddings_cpu,
            EMBEDDINGS_FILE,
        )

        self.chunks = all_chunks

        self.document_embeddings = (
            embeddings.to(
                self.device
            )
        )

        return {
            "documents":
            len(pdf_files),

            "chunks":
            len(all_chunks),

            "embedding_shape":
            list(
                embeddings.shape
            ),
        }

    def search(
        self,
        query,
        retrieval_top_k=10,
        result_top_k=5,
        allowed_filenames=None,
    ):

        if not self.ready:
            raise RuntimeError(
                "Search service is not ready."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        start_time = (
            time.perf_counter()
        )

        candidate_indices = None

        if allowed_filenames is not None:
            allowed_filename_keys = {
                str(filename).casefold()
                for filename in allowed_filenames
            }

            if not allowed_filename_keys:
                return {
                    "query": query,
                    "latency_ms": round(
                        (
                            time.perf_counter()
                            - start_time
                        ) * 1000,
                        2,
                    ),
                    "results": [],
                }

        if (
            not self.chunks
            or self.document_embeddings is None
        ):
            raise RuntimeError(
                "Document index is empty."
            )

        candidate_embeddings = (
            self.document_embeddings
        )

        if allowed_filenames is not None:
            candidate_indices = [
                index
                for index, chunk
                in enumerate(self.chunks)
                if str(
                    chunk.get("filename", "")
                ).casefold()
                in allowed_filename_keys
            ]

            if not candidate_indices:
                return {
                    "query": query,
                    "latency_ms": round(
                        (
                            time.perf_counter()
                            - start_time
                        ) * 1000,
                        2,
                    ),
                    "results": [],
                }

            index_tensor = torch.tensor(
                candidate_indices,
                device=(
                    self.document_embeddings.device
                ),
                dtype=torch.long,
            )

            candidate_embeddings = (
                self.document_embeddings.index_select(
                    0,
                    index_tensor,
                )
            )

        query_embedding = (
            self.embedding_model.encode(
                query,
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
        )

        query_embedding = (
            query_embedding.to(
                self.device
            )
        )

        retrieval_scores = (
            query_embedding
            @ candidate_embeddings.T
        )

        retrieval_top_k = min(
            retrieval_top_k,
            len(candidate_embeddings),
        )

        top_indices = torch.topk(
            retrieval_scores,
            k=retrieval_top_k,
        ).indices.tolist()

        candidates = []

        for candidate_index in top_indices:

            index = (
                candidate_index
                if candidate_indices is None
                else candidate_indices[
                    candidate_index
                ]
            )

            chunk = dict(
                self.chunks[index]
            )

            chunk[
                "retrieval_score"
            ] = float(
                retrieval_scores[
                    candidate_index
                ].item()
            )

            candidates.append(
                chunk
            )

        pairs = [
            (
                query,
                candidate["text"],
            )
            for candidate
            in candidates
        ]

        reranker_scores = (
            self.reranker.predict(
                pairs
            )
        )

        ranked = sorted(
            zip(
                candidates,
                reranker_scores,
            ),
            key=lambda item:
            float(item[1]),
            reverse=True,
        )

        result_top_k = min(
            result_top_k,
            len(ranked),
        )

        results = []

        for rank, (
            candidate,
            reranker_score,
        ) in enumerate(
            ranked[:result_top_k],
            1,
        ):

            results.append(
                {
                    "rank":
                    rank,

                    "filename":
                    candidate[
                        "filename"
                    ],

                    "page":
                    candidate[
                        "page"
                    ],

                    "chunk_id":
                    candidate[
                        "chunk_id"
                    ],

                    "section_id":
                    candidate.get(
                        "section_id"
                    ),

                    "text":
                    candidate[
                        "text"
                    ],

                    "retrieval_score":
                    round(
                        candidate[
                            "retrieval_score"
                        ],
                        4,
                    ),

                    "reranker_score":
                    round(
                        float(
                            reranker_score
                        ),
                        4,
                    ),
                }
            )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        return {
            "query":
            query,

            "latency_ms":
            round(
                latency_ms,
                2,
            ),

            "results":
            results,
        }

    def get_documents(self):

        documents = {}

        for chunk in self.chunks:

            filename = chunk[
                "filename"
            ]

            if filename not in documents:
                documents[
                    filename
                ] = {
                    "filename":
                    filename,

                    "chunks":
                    0,

                    "pages":
                    set(),
                }

            documents[
                filename
            ]["chunks"] += 1

            documents[
                filename
            ]["pages"].add(
                chunk["page"]
            )

        output = []

        for document in documents.values():

            output.append(
                {
                    "filename":
                    document[
                        "filename"
                    ],

                    "chunks":
                    document[
                        "chunks"
                    ],

                    "pages":
                    len(
                        document[
                            "pages"
                        ]
                    ),
                }
            )

        return output


    def get_document_path(
        self,
        filename,
    ):
        filename = (
            filename or ""
        ).strip()

        if not filename:
            raise ValueError(
                "Filename is required."
            )

        safe_name = Path(
            filename
        ).name

        if (
            safe_name != filename
            or Path(
                safe_name
            ).suffix.lower()
            != ".pdf"
        ):
            raise ValueError(
                "Invalid PDF filename."
            )

        document_path = (
            DOCUMENTS_DIR
            / safe_name
        )

        if (
            not document_path.exists()
            or not document_path.is_file()
        ):
            raise FileNotFoundError(
                f"Document not found: {safe_name}"
            )

        return document_path


    def clear_index(self):

        self.chunks = []

        self.document_embeddings = (
            None
        )

        CHUNKS_FILE.unlink(
            missing_ok=True
        )

        EMBEDDINGS_FILE.unlink(
            missing_ok=True
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def delete_document(
        self,
        filename,
    ):
        document_path = (
            self.get_document_path(
                filename
            )
        )

        deleted_filename = (
            document_path.name
        )

        document_path.unlink()

        result = (
            self.build_index()
        )

        return {
            "deleted":
            deleted_filename,

            **result,
        }

    def cleanup(self):

        self.embedding_model = None
        self.reranker = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
