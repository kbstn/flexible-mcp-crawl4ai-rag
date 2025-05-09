"""configurable embedding dimension support

Revision ID: c39dbe1f87f7
Revises: 2da3a45566c4
Create Date: 2025-05-09 10:53:40 # Timestamp of generation

"""
from typing import Sequence, Union, Optional
import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text as sql_text
from pgvector.sqlalchemy import Vector

# Attempt to import settings from the application
try:
    from src.utils import settings as app_settings
    current_settings = app_settings
except ImportError:
    class FallbackSettings:
        _dim_env_val = os.getenv('OLLAMA_EMBEDDING_DIM')
        OLLAMA_EMBEDDING_DIM: int = int(_dim_env_val) if _dim_env_val and _dim_env_val.isdigit() else 1024
    current_settings = FallbackSettings()
    print("Warning: Could not import 'src.utils.settings'. Falling back to OS environment for OLLAMA_EMBEDDING_DIM.")

# revision identifiers, used by Alembic.
revision: str = 'c39dbe1f87f7'
down_revision: Union[str, None] = '2da3a45566c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = 'crawledpage'
COLUMN_NAME = 'embedding'
DEFAULT_FALLBACK_DIMENSION_IF_NOT_FOUND = 768

def get_current_vector_dimension(conn: sa.engine.Connection, table_name: str, column_name: str) -> Optional[int]:
    query = sql_text(f"""
        SELECT atttypmod
        FROM pg_attribute att
        JOIN pg_class cla ON cla.oid = att.attrelid
        JOIN pg_namespace nsp ON nsp.oid = cla.relnamespace
        JOIN pg_type typ ON typ.oid = att.atttypid
        WHERE cla.relname = :table_name
          AND att.attname = :column_name
          AND typ.typname = 'vector';
    """)
    result = conn.execute(query, {'table_name': table_name, 'column_name': column_name}).scalar_one_or_none()
    if result is not None and result > 0:
        return int(result)
    print(f"Warning: Could not dynamically determine current dimension for {table_name}.{column_name}.")
    return None


def upgrade() -> None:
    conn = op.get_bind()
    target_dimension = current_settings.OLLAMA_EMBEDDING_DIM
    existing_dimension = get_current_vector_dimension(conn, TABLE_NAME, COLUMN_NAME)

    if existing_dimension is None:
        print(f"Warning: Could not determine existing dimension for {TABLE_NAME}.{COLUMN_NAME}. Assuming fallback {DEFAULT_FALLBACK_DIMENSION_IF_NOT_FOUND}.")
        existing_dimension_for_alter = DEFAULT_FALLBACK_DIMENSION_IF_NOT_FOUND
    else:
        print(f"Detected existing dimension for {TABLE_NAME}.{COLUMN_NAME} as {existing_dimension}.")
        existing_dimension_for_alter = existing_dimension

    if existing_dimension_for_alter == target_dimension:
        print(f"Target dimension ({target_dimension}) is same as existing ({existing_dimension_for_alter}). No alteration needed.")
        return

    print(f"Attempting to alter '{TABLE_NAME}.{COLUMN_NAME}' from VECTOR({existing_dimension_for_alter}) to VECTOR({target_dimension}) using NULL for existing data.")
    
    # Use raw SQL with USING NULL to alter the column type
    op.execute(f"""
        ALTER TABLE {TABLE_NAME}
        ALTER COLUMN {COLUMN_NAME} TYPE VECTOR({target_dimension})
        USING NULL;
    """)

    print(f"'{TABLE_NAME}.{COLUMN_NAME}' column altered to VECTOR({target_dimension}). Existing data set to NULL.")


def downgrade() -> None:
    conn = op.get_bind()
    dimension_being_downgraded = get_current_vector_dimension(conn, TABLE_NAME, COLUMN_NAME)

    if dimension_being_downgraded is None:
        print(f"Warning: Could not determine current dimension for {TABLE_NAME}.{COLUMN_NAME} during downgrade.")
        # Fallback to what the current settings indicate the dimension *should* be
        dimension_being_downgraded = current_settings.OLLAMA_EMBEDDING_DIM

    downgrade_to_dimension = DEFAULT_FALLBACK_DIMENSION_IF_NOT_FOUND

    if dimension_being_downgraded == downgrade_to_dimension:
        print(f"Current dimension ({dimension_being_downgraded}) is already downgrade target ({downgrade_to_dimension}). No alteration.")
        return

    print(f"Attempting to alter '{TABLE_NAME}.{COLUMN_NAME}' from VECTOR({dimension_being_downgraded}) to VECTOR({downgrade_to_dimension}) using NULL for existing data.")

    # Use raw SQL with USING NULL for downgrade as well
    op.execute(f"""
        ALTER TABLE {TABLE_NAME}
        ALTER COLUMN {COLUMN_NAME} TYPE VECTOR({downgrade_to_dimension})
        USING NULL;
    """)
               
    # Removed op.comment_on_column as it caused an AttributeError
    print(f"'{TABLE_NAME}.{COLUMN_NAME}' column altered back to VECTOR({downgrade_to_dimension}). Existing data set to NULL.")
