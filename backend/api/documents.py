"""
Documents API - File upload, ingestion, and RAG retrieval
"""

import os
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from backend.db.database import get_db
from backend.core.security import get_current_user
from backend.core.config import settings
from backend.models.models import User, Document
from backend.schemas.schemas import DocumentResponse, RAGQueryRequest
from backend.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".rtf", ".txt", ".pdf", ".docx", ".md", ".csv", ".py", ".js", ".json"}


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document and ingest it into the vector store."""
    # Validate file
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Save file to disk
    safe_name = f"{uuid.uuid4().hex}{ext}"
    user_dir = os.path.join(settings.UPLOAD_DIR, str(current_user.user_id))
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, safe_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    doc = Document(
        user_id=current_user.user_id,
        filename=safe_name,
        original_filename=file.filename or safe_name,
        file_path=file_path,
        file_size=len(content),
        file_type=ext.lstrip("."),
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Ingest into vector store (async background-style)
    try:
        chunk_count = await rag_service.ingest_document(db, doc)
        logger.info(f"Document {doc.document_id} ingested: {chunk_count} chunks")
    except Exception as e:
        logger.error(f"Ingestion failed for doc {doc.document_id}: {e}")

    return DocumentResponse.model_validate(doc)


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents uploaded by the current user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.user_id)
        .order_by(Document.upload_date.desc())
    )
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get metadata for a specific document."""
    result = await db.execute(
        select(Document).where(
            Document.document_id == document_id,
            Document.user_id == current_user.user_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and remove its embeddings."""
    result = await db.execute(
        select(Document).where(
            Document.document_id == document_id,
            Document.user_id == current_user.user_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove file from disk
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception as e:
        logger.warning(f"Could not delete file: {e}")

    await db.delete(doc)


@router.post("/query/rag")
async def rag_query(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Query documents using RAG - returns relevant text chunks."""
    chunks = await rag_service.retrieve_context(
        db,
        current_user.user_id,
        request.query,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )
    return {
        "query": request.query,
        "chunks": chunks,
        "total_chunks": len(chunks),
    }
