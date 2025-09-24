"""create_initial_tables_and_seed_

Revision ID: a1766463d8d4
Revises: 
Create Date: 2025-09-02 17:35:35.501272

"""
from typing import Sequence, Union

from pgvector.sqlalchemy import VECTOR
from alembic import op
import sqlalchemy as sa
import os
import logging
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1766463d8d4'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger(f"alembic.revision.{revision}")


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Créer l'extension vector et le type ENUM s'ils n'existent pas
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    posture_enum = postgresql.ENUM('assis', 'debout', 'a_pieds', name='posture_enum', create_type=False)
    posture_enum.create(op.get_bind(), checkfirst=True)

    # 2. Créer toutes les tables
    op.create_table('dogs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('breed', sa.Text(), nullable=True),
        sa.Column('owner_name', sa.Text(), nullable=True)
    )

    op.create_table('embeddings',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', VECTOR(384), nullable=False),
    )

    op.create_table('reference_posture_videos',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('posture', posture_enum, nullable=False),
        sa.Column('video_path', sa.Text(), nullable=False)
    )

    op.create_table('video_sessions',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('dog_id', sa.Integer(), nullable=False),
        sa.Column('posture', posture_enum, nullable=False),
        sa.Column('session_start', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('session_end', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('total_frames_processed', sa.Integer(), server_default='0', nullable=True),
        sa.Column('success_detected', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.ForeignKeyConstraint(['dog_id'], ['dogs.id'], name='fk_video_sessions_dog_id')
    )

    op.create_table('posture_detection_results',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('posture', posture_enum, nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('result', sa.Text(), nullable=False),
        sa.Column('timestamp', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['video_sessions.id'], name='fk_pdr_session_id'),
    )

    op.create_table('validated_postures',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('dog_id', sa.Integer(), nullable=False),
        sa.Column('posture', posture_enum, nullable=False),
        sa.Column('validated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True)
    )

    # 3. Mettre à jour les tables existantes
    # Table 'dogs'

    # Table 'posture_detection_results'
    # Les opérations 'add_column' et 'alter_column' ne sont plus nécessaires car les tables
    # sont créées directement avec la bonne structure.

    # 4. Ajouter les contraintes de clé étrangère maintenant que toutes les tables et colonnes sont en place
    op.create_foreign_key('fk_validated_postures_dog_id', 'validated_postures', 'dogs', ['dog_id'], ['id'])
    op.create_foreign_key('fk_pdr_video_id', 'posture_detection_results', 'reference_posture_videos', ['video_id'], ['id'])

    # 5. Seeding des données de référence
    seed_reference_videos()


def seed_reference_videos():
    """
    Remplit la table reference_posture_videos avec les vidéos du dossier static/videos.
    Cette fonction est conçue pour être appelée depuis une migration Alembic.
    """
    # Définition de la table pour l'insertion de données brutes (bulk insert)
    ref_videos_table = sa.table('reference_posture_videos',
        sa.column('posture', sa.String),
        sa.column('video_path', sa.Text)
    )

    # Le chemin est relatif à la racine du projet où alembic est exécuté
    STATIC_VIDEOS_DIR = "static/videos"
    if not os.path.isdir(STATIC_VIDEOS_DIR):
        logger.warning(f"Le dossier des vidéos de référence '{STATIC_VIDEOS_DIR}' n'a pas été trouvé. Le seeding est ignoré.")
        return

    videos_to_insert = []
    for filename in os.listdir(STATIC_VIDEOS_DIR):
        if filename.endswith(".mp4"):
            try:
                # Normalise le nom du fichier (supprime espaces superflus, remplace par '_')
                normalized_filename = "_".join(filename.strip().split())
                base_name = normalized_filename.removesuffix('.mp4').strip()
                posture_part = base_name.rsplit('_', 1)[0].strip()
                posture_name = posture_part.replace("chien_", "").replace("chiens_", "").strip()
                
                video_path = f"static/videos/{normalized_filename}"
                videos_to_insert.append({'posture': posture_name, 'video_path': video_path})
            except (ValueError, IndexError):
                logger.warning(f"Impossible de traiter le fichier vidéo '{filename}'. Le nom ne correspond pas au format attendu.")

    if videos_to_insert:
        op.bulk_insert(ref_videos_table, videos_to_insert)
        logger.info(f"{len(videos_to_insert)} vidéos de référence insérées dans la base de données.")


def downgrade() -> None:
    """Downgrade schema."""
    # Inversion des opérations de 'upgrade'
    op.drop_constraint('fk_pdr_video_id', 'posture_detection_results', type_='foreignkey')
    op.drop_constraint('fk_validated_postures_dog_id', 'validated_postures', type_='foreignkey')

    # L'ordre de suppression est l'inverse de la création
    op.drop_table('validated_postures')
    op.drop_table('posture_detection_results')
    op.drop_table('video_sessions')
    op.drop_table('reference_posture_videos')
    op.drop_table('embeddings')
    op.drop_table('dogs')

    # Supprimer le type ENUM
    posture_enum = postgresql.ENUM('assis', 'debout', 'a_pieds', name='posture_enum', create_type=False)
    posture_enum.drop(op.get_bind(), checkfirst=True)