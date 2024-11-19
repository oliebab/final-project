from app import create_app
from sqlalchemy import text

app = create_app()

with app.app_context():
    from app.database import db
    try:
        result = db.session.execute(text("SELECT 1")).scalar()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {str(e)}")