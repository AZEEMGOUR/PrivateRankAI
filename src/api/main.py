from contextlib import asynccontextmanager
import logging
from pathlib import Path
from uuid import uuid4
from fastapi.responses import FileResponse
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    Field,
)

from src.api.search_service import (
    PrivateRankSearchService,
)
from src.ingestion.pdf_loader import (
    load_pdf_pages,
)

from src.auth.dependencies import (
    authenticate_user,
    authentication_error,
    get_current_user,
    user_store,
)
from src.auth.security import (
    create_access_token,
    get_access_token_expire_minutes,
    warn_if_using_development_secret,
)
from src.storage.document_store import (
    DocumentStore,
)


search_service = PrivateRankSearchService()
document_store = DocumentStore()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "real_documents"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

logger = logging.getLogger(__name__)


def normalize_tags(tags):
    if not tags:
        return []

    normalized = []
    seen = set()

    for value in tags.split(","):
        tag = value.strip()
        key = tag.casefold()

        if tag and key not in seen:
            normalized.append(tag)
            seen.add(key)

    return normalized


def restore_document_after_failed_delete(
    staged_path,
    document_path,
):
    try:
        if (
            staged_path.exists()
            and not document_path.exists()
        ):
            staged_path.replace(
                document_path
            )

        search_service.build_index()

    except Exception:
        logger.exception(
            "Unable to restore document/index after failed deletion."
        )


def rollback_new_upload(
    destination,
    document_id,
):
    try:
        if destination.exists():
            destination.unlink()
    except OSError:
        logger.exception(
            "Unable to remove PDF during upload rollback."
        )

    try:
        document_store.delete_document_metadata(
            document_id
        )
    except Exception:
        logger.exception(
            "Unable to remove metadata during upload rollback."
        )

    try:
        search_service.build_index()
    except Exception:
        logger.exception(
            "Unable to restore index during upload rollback."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    document_store.initialize_database()
    user_store.initialize_database()
    document_store.bootstrap_existing_pdfs(
        DOCUMENTS_DIR
    )
    warn_if_using_development_secret()

    search_service.load()

    yield

    search_service.cleanup()


app = FastAPI(
    title="PrivateRankAI API",
    version="0.1.0",
    description=(
        "Private enterprise document "
        "retrieval and reranking API."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
    )

    retrieval_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    result_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    department: str | None = Field(
        default=None,
        max_length=200,
    )

    category: str | None = Field(
        default=None,
        max_length=200,
    )


class LoginRequest(BaseModel):
    email: str = Field(
        min_length=3,
        max_length=320,
    )

    password: str = Field(
        min_length=1,
        max_length=1024,
    )


@app.post("/auth/login")
def login(request: LoginRequest):
    user = authenticate_user(
        request.email,
        request.password,
    )

    if user is None:
        raise authentication_error()

    expire_minutes = (
        get_access_token_expire_minutes()
    )

    return {
        "access_token": create_access_token(
            user["id"]
        ),
        "token_type": "bearer",
        "expires_in": expire_minutes * 60,
        "user": user,
    }


@app.get("/auth/me")
def auth_me(
    current_user=Depends(
        get_current_user
    ),
):
    return current_user


@app.get("/")
def root():
    return {
        "name": "PrivateRankAI",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": search_service.ready,
        "device": search_service.device,
        "indexed_chunks": len(search_service.chunks),
    }


@app.get("/documents")
def documents():
    index_documents = {
        item["filename"].casefold(): item
        for item in search_service.get_documents()
    }

    output = []

    for document in document_store.list_documents():
        index_document = index_documents.get(
            document["filename"].casefold(),
            {},
        )

        output.append(
            {
                **document,
                "pages": index_document.get(
                    "pages",
                    0,
                ),
                "chunks": index_document.get(
                    "chunks",
                    0,
                ),
            }
        )

    return {
        "documents": output
    }


@app.post("/documents/index")
def rebuild_index():
    try:
        result = search_service.build_index()

        return {
            "status": "success",
            **result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error



@app.get(
    "/documents/{filename}/file"
)
def open_document(
    filename: str,
):
    try:
        document_path = (
            search_service
            .get_document_path(
                filename
            )
        )

        return FileResponse(
            path=str(
                document_path
            ),
            media_type=(
                "application/pdf"
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.delete(
    "/documents/{filename}"
)
def delete_document(
    filename: str,
):
    staged_path = None
    document_path = None

    try:
        document_path = (
            search_service
            .get_document_path(
                filename
            )
        )

        metadata = (
            document_store
            .get_document_by_filename(
                document_path.name
            )
        )

        staged_path = document_path.with_name(
            f".{document_path.name}."
            f"{uuid4().hex}.deleting"
        )

        document_path.replace(
            staged_path
        )

        result = search_service.build_index()

        if metadata is not None:
            deleted = (
                document_store
                .delete_document_metadata(
                    metadata["id"]
                )
            )

            if not deleted:
                raise RuntimeError(
                    "Document metadata was not deleted."
                )

        try:
            staged_path.unlink()
        except OSError:
            logger.exception(
                "Document was removed from the application, "
                "but its staged file could not be deleted."
            )

        return {
            "status":
            "success",

            "deleted":
            document_path.name,

            **result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        if (
            staged_path is not None
            and document_path is not None
        ):
            restore_document_after_failed_delete(
                staged_path,
                document_path,
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to delete document safely. "
                "The original document was restored when possible."
            ),
        ) from error


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    display_name: str | None = Form(None),
    department: str | None = Form(None),
    category: str | None = Form(None),
    tags: str | None = Form(None),
    description: str | None = Form(None),
):
    original_name = Path(
        file.filename or ""
    ).name.strip()

    if not original_name:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if Path(original_name).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported right now.",
        )

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = DOCUMENTS_DIR / original_name

    if destination.exists():
        stem = destination.stem
        suffix = destination.suffix
        counter = 2

        while destination.exists():
            destination = (
                DOCUMENTS_DIR
                / f"{stem}_{counter}{suffix}"
            )
            counter += 1

    total_size = 0

    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF is larger than 50 MB.",
                    )

                output_file.write(chunk)

    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    finally:
        await file.close()

    try:
        pages = load_pdf_pages(destination)

        readable_pages = [
            page
            for page in pages
            if page.get("text", "").strip()
        ]

        if not readable_pages:
            raise ValueError(
                "PDF contains no readable text."
            )

    except Exception as error:
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read PDF: {error}",
        ) from error

    try:
        document = document_store.create_document(
            filename=destination.name,
            display_name=(
                (display_name or "").strip()
                or Path(original_name).stem
            ),
            department=(
                (department or "").strip()
                or "General"
            ),
            category=(
                (category or "").strip()
                or "Other"
            ),
            description=(
                description or ""
            ).strip(),
            tags=normalize_tags(tags),
            status="indexed",
        )

    except Exception as error:
        try:
            if destination.exists():
                destination.unlink()
        except OSError:
            logger.exception(
                "Unable to remove PDF after metadata creation failed."
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save document metadata. "
                "The uploaded PDF was removed."
            ),
        ) from error

    try:
        index_result = search_service.build_index()
    except Exception as error:
        rollback_new_upload(
            destination,
            document["id"],
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document indexing failed. "
                "The new PDF and metadata were rolled back."
            ),
        ) from error

    return {
        "status": "success",
        "filename": destination.name,
        "size_bytes": total_size,
        "pages": len(readable_pages),
        "document": document,
        "index": index_result,
    }


@app.post("/search")
def search(
    request: SearchRequest,
):
    try:
        department = (
            request.department or ""
        ).strip() or None
        category = (
            request.category or ""
        ).strip() or None

        allowed_filenames = None

        if department or category:
            allowed_filenames = (
                document_store
                .list_filenames_by_metadata(
                    department=department,
                    category=category,
                )
            )

        return search_service.search(
            query=request.query,
            retrieval_top_k=(
                request.retrieval_top_k
            ),
            result_top_k=(
                request.result_top_k
            ),
            allowed_filenames=(
                allowed_filenames
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
