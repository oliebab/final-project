# config.py
class Config:
    # Add connection pool settings
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:Keklol360**@localhost/diggerz?charset=utf8mb4'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_timeout': 30,
        'pool_pre_ping': True
    }
    PER_PAGE = 10