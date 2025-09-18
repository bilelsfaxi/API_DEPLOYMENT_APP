import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.pool import StaticPool
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.database import Base
from api import crud, models
from api.schemas import db_schemas
from fastapi import HTTPException

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest_asyncio.fixture(scope="function")
async def db_session():
    tables_to_create = [
        table for table in Base.metadata.sorted_tables if table.name != 'embeddings'
    ]
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=tables_to_create)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        await db.close()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture(scope="function")
async def seed_data(db_session):
    videos_to_seed = []
    for posture in models.PostureEnum:
        for i in range(1, 5):
            videos_to_seed.append(models.ReferencePostureVideo(
                posture=posture, video_path=f"/fake/{posture.value}_{i}.mp4"
            ))
    db_session.add_all(videos_to_seed)
    dog_to_seed = models.Dog(name="Test Dog", breed="Tester")
    db_session.add(dog_to_seed)
    await db_session.commit()

    dog_result = await db_session.execute(select(models.Dog).filter_by(name="Test Dog"))
    live_dog = dog_result.scalar_one()
    videos_result = await db_session.execute(select(models.ReferencePostureVideo))
    live_videos = videos_result.scalars().all()
    video_data = {
        p: sorted([v.id for v in live_videos if v.posture == p])
        for p in models.PostureEnum
    }
    return {"dog_id": live_dog.id, "videos": video_data}

@pytest.mark.asyncio
async def test_create_session(db_session, seed_data):
    dog_id = seed_data["dog_id"]
    session_data = db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum.assis)
    session = await crud.create_video_session(db=db_session, session=session_data)
    assert session is not None
    assert session.dog_id == dog_id
    assert session.posture == models.PostureEnum.assis
    assert not session.success_detected

@pytest.mark.asyncio
async def test_less_than_4_successes(db_session, seed_data):
    dog_id = seed_data["dog_id"]
    session_data = db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum.assis)
    session = await crud.create_video_session(db=db_session, session=session_data)
    video_id = seed_data["videos"][models.PostureEnum.assis][0]
    for _ in range(3):
        attempt = db_schemas.PostureAttemptCreate(
            session_id=session.id,
            video_id=video_id,
            confidence=0.9,
            result='success',
            prediction_time=0.5,
            frames_processed=1
        )
        await crud.create_posture_attempt(db=db_session, attempt=attempt)
    refreshed_session = await crud.get_session_by_id(db=db_session, session_id=session.id)
    assert not refreshed_session.success_detected
    assert refreshed_session.total_frames_processed == 3

@pytest.mark.asyncio
async def test_exactly_4_successes(db_session, seed_data):
    dog_id = seed_data["dog_id"]
    session_data = db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum.debout)
    session = await crud.create_video_session(db=db_session, session=session_data)
    debout_video_ids = seed_data["videos"][models.PostureEnum.debout]
    for i in range(4):
        attempt = db_schemas.PostureAttemptCreate(
            session_id=session.id,
            video_id=debout_video_ids[i],
            confidence=0.95,
            result='success',
            prediction_time=0.5,
            frames_processed=1
        )
        await crud.create_posture_attempt(db=db_session, attempt=attempt)
    refreshed_session = await crud.get_session_by_id(db=db_session, session_id=session.id)
    assert refreshed_session.success_detected
    assert refreshed_session.total_frames_processed == 4
    validated_postures = await crud.get_validated_postures_by_dog(db=db_session, dog_id=dog_id)
    assert len(validated_postures) == 1
    assert validated_postures[0].posture == models.PostureEnum.debout

@pytest.mark.asyncio
async def test_video_selection_is_deterministic(db_session, seed_data):
    dog_id = seed_data["dog_id"]
    session_data = db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum.a_pieds)
    session = await crud.create_video_session(db=db_session, session=session_data)
    all_video_ids = set(seed_data["videos"][models.PostureEnum.a_pieds])
    used_video_ids = set()
    videos = await crud.get_next_videos_for_session(db=db_session, session_id=session.id)
    assert len(videos) <= 4
    for video in videos:
        assert video.id in all_video_ids
        assert video.id not in used_video_ids
        used_video_ids.add(video.id)
        await crud.create_posture_attempt(db=db_session, attempt=db_schemas.PostureAttemptCreate(
            session_id=session.id,
            video_id=video.id,
            confidence=0.9,
            result='fail',
            prediction_time=0.5,
            frames_processed=1
        ))

@pytest.mark.asyncio
async def test_error_on_mismatched_posture(db_session, seed_data):
    dog_id = seed_data["dog_id"]
    session_data = db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum.assis)
    session = await crud.create_video_session(db=db_session, session=session_data)
    debout_video_id = seed_data["videos"][models.PostureEnum.debout][0]
    with pytest.raises(HTTPException) as excinfo:
        await crud.create_posture_attempt(db=db_session, attempt=db_schemas.PostureAttemptCreate(
            session_id=session.id,
            video_id=debout_video_id,
            confidence=0.9,
            result='success',
            prediction_time=0.5,
            frames_processed=1
        ))
    assert excinfo.value.status_code == 400
    assert "does not match session posture" in excinfo.value.detail

@pytest.mark.asyncio
async def test_error_on_invalid_session_id(db_session, seed_data):
    video_id = seed_data["videos"][models.PostureEnum.assis][0]
    invalid_session_id = 9999
    with pytest.raises(HTTPException) as excinfo:
        await crud.create_posture_attempt(db=db_session, attempt=db_schemas.PostureAttemptCreate(
            session_id=invalid_session_id,
            video_id=video_id,
            confidence=0.9,
            result='success',
            prediction_time=0.5,
            frames_processed=1
        ))
    assert excinfo.value.status_code == 404
    assert "Session with id 9999 not found" in excinfo.value.detail

@pytest.mark.asyncio
async def test_error_on_invalid_video_id(db_session, seed_data):
    dog_id = seed_data["dog_id"]
    session_data = db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum.assis)
    session = await crud.create_video_session(db=db_session, session=session_data)
    invalid_video_id = 9999
    with pytest.raises(HTTPException) as excinfo:
        await crud.create_posture_attempt(db=db_session, attempt=db_schemas.PostureAttemptCreate(
            session_id=session.id,
            video_id=invalid_video_id,
            confidence=0.9,
            result='success',
            prediction_time=0.5,
            frames_processed=1
        ))
    assert excinfo.value.status_code == 404
    assert "Reference video with id 9999 not found" in excinfo.value.detail