from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import AnyStr, Dict


Base = declarative_base()


def init_db(db_path: Path) -> Session:
    """
    Initialize a SQLite3 database located at `db_path` and return a Session

    :param db_path: The path to create the DB at
    :return: Session object for the database connection
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


class SampleData(Base):
    """
    An ORM to track/manage data extracted from binary files
    """
    __tablename__ = 'sample'

    sha256 = Column(String(64), primary_key=True)
    # The 7 Adobe predictors
    debug_size = Column(Integer)
    image_version = Column(Integer)
    import_rva = Column(Integer)
    export_size = Column(Integer)
    resource_size = Column(Integer)
    num_sections = Column(Integer)
    virtual_size_2 = Column(Integer)

    def __repr__(self) -> AnyStr:
        return f"<SampleData {self.sha256}>"

    def __str__(self) -> AnyStr:
        return repr(self)

    def serialize(self) -> Dict:
        return {'sha256': self.sha256}
