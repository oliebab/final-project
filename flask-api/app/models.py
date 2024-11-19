from .database import db
from datetime import datetime

class Artist(db.Model):
    artist_id = db.Column(db.String(255), primary_key=True)
    artist_name = db.Column(db.String(255), nullable=False)
    updated_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Track(db.Model):
    track_id = db.Column(db.String(255), primary_key=True)
    track_title = db.Column(db.String(255), nullable=False)
    duration_ms = db.Column(db.Integer)
    isrc = db.Column(db.String(255), unique=True)
    track_number = db.Column(db.Integer)
    release_id = db.Column(db.String(255), db.ForeignKey('release.release_id'))
    explicit = db.Column(db.Boolean, default=False)
    disc_number = db.Column(db.Integer)
    preview_url = db.Column(db.Text)
    updated_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Release(db.Model):
    release_id = db.Column(db.String(255), primary_key=True)
    release_title = db.Column(db.String(255), nullable=False)
    release_date = db.Column(db.Date)
    upc = db.Column(db.String(255))
    popularity = db.Column(db.Integer)
    total_tracks = db.Column(db.Integer)
    album_type = db.Column(db.String(50))
    release_img = db.Column(db.String(255))
    label_name = db.Column(db.String(255))
    updated_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class AudioFeatures(db.Model):
    isrc = db.Column(db.String(255), db.ForeignKey('track.isrc'), primary_key=True)
    acousticness = db.Column(db.Float)
    danceability = db.Column(db.Float)
    energy = db.Column(db.Float)
    instrumentalness = db.Column(db.Float)
    key = db.Column(db.Integer)
    liveness = db.Column(db.Float)
    loudness = db.Column(db.Float)
    mode = db.Column(db.Integer)
    speechiness = db.Column(db.Float)
    tempo = db.Column(db.Float)
    time_signature = db.Column(db.Integer)
    valence = db.Column(db.Float)
    updated_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)