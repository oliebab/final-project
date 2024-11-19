from flask import Blueprint, request, jsonify
from .database import db
from sqlalchemy import text
from flask import render_template_string
from flask import Blueprint
import contextlib

bp = Blueprint('api', __name__, url_prefix='/api')

def get_db_connection():
    """Get a fresh database connection"""
    return db.engine.connect()

# In routes.py

@bp.route('/', methods=['GET', 'POST'])
def index():
    """API Documentation and Interactive Forms with Refined Filters"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diggerz API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background-color: #f4f4f9; 
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }
            .container {
                width: 80%;
                max-width: 1200px;
                background-color: #ffffff;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }
            .logo {
                max-width: 100px;
                margin: 0 auto 20px auto;
            }
            h1 {
                color: #333333;
                margin-bottom: 20px;
            }
            .forms-container {
                display: flex;
                justify-content: space-between;
            }
            .form-section {
                width: 48%;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.05);
            }
            .form-group {
                margin-bottom: 15px;
                text-align: left;
            }
            label {
                font-weight: bold;
                color: #555555;
            }
            input[type="text"], input[type="number"], input[type="date"], select {
                width: 100%;
                padding: 8px;
                margin-top: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="/static/image.png" alt="Logo" class="logo">
            <h1>Diggerz API</h1>
            <div class="forms-container">
                <div class="form-section">
                    <h2>Artists</h2>
                    <form action="/api/artists" method="get">
                        <div class="form-group">
                            <label for="search">Search Artists:</label>
                            <input type="text" name="search" id="search" placeholder="Enter artist name">
                        </div>
                        <div class="form-group">
                            <label for="min_tracks">Minimum Track Count:</label>
                            <input type="number" name="min_tracks" id="min_tracks" value="1" min="1">
                        </div>
                        <div class="form-group">
                            <label for="sort_by">Sort By:</label>
                            <select name="sort_by" id="sort_by">
                                <option value="name">Name</option>
                                <option value="track_count">Track Count</option>
                                <option value="release_count">Release Count</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="order">Order:</label>
                            <select name="order" id="order">
                                <option value="asc">Ascending</option>
                                <option value="desc">Descending</option>
                            </select>
                        </div>
                        <button type="submit">Search Artists</button>
                    </form>
                </div>
                
                <div class="form-section">
                    <h2>Tracks</h2>
                    <form action="/api/tracks" method="get">
                        <div class="form-group">
                            <label for="search">Search Tracks:</label>
                            <input type="text" name="search" id="search" placeholder="Enter track name">
                        </div>
                        <div class="form-group">
                            <label for="label">Label:</label>
                            <input type="text" name="label" id="label" placeholder="Enter label name">
                        </div>
                        <div class="form-group">
                            <label for="min_danceability">Minimum Danceability:</label>
                            <input type="number" name="min_danceability" id="min_danceability" step="0.1" min="0" max="1">
                        </div>
                        <div class="form-group">
                            <label for="min_energy">Minimum Energy:</label>
                            <input type="number" name="min_energy" id="min_energy" step="0.1" min="0" max="1">
                        </div>
                        <div class="form-group">
                            <label for="min_tempo">Minimum Tempo:</label>
                            <input type="number" name="min_tempo" id="min_tempo" placeholder="e.g., 60 BPM">
                        </div>
                        <div class="form-group">
                            <label for="max_tempo">Maximum Tempo:</label>
                            <input type="number" name="max_tempo" id="max_tempo" placeholder="e.g., 200 BPM">
                        </div>
                        <div class="form-group">
                            <label for="num_tracks">Number of Tracks to Display:</label>
                            <input type="number" name="num_tracks" id="num_tracks" value="10" min="1">
                        </div>
                        <div class="form-group">
                            <label for="sort_by">Sort By:</label>
                            <select name="sort_by" id="sort_by">
                                <option value="danceability">Danceability</option>
                                <option value="energy">Energy</option>
                                <option value="valence">Valence</option>
                                <option value="tempo">Tempo</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="order">Order:</label>
                            <select name="order" id="order">
                                <option value="asc">Ascending</option>
                                <option value="desc">Descending</option>
                            </select>
                        </div>
                        <button type="submit">Search Tracks</button>
                    </form>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')



@bp.route('/artists', methods=['GET'])
def get_artists():
    """
    Get list of artists with filtering and sorting options.
    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 10)
    - sort_by: Sort field (name, tracks, popularity) (default: name)
    - order: Sort order (asc, desc) (default: asc)
    - min_tracks: Minimum number of tracks (default: 0)
    - search: Search in artist name
    """
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        sort_by = request.args.get('sort_by', 'name')
        order = request.args.get('order', 'asc').upper()
        min_tracks = request.args.get('min_tracks', 0, type=int)
        search = request.args.get('search', '')

        # Validate order parameter
        if order not in ['ASC', 'DESC']:
            order = 'ASC'

        with db.engine.connect() as conn:
            # Build base query
            base_query = """
                WITH artist_stats AS (
                    SELECT 
                        a.artist_id,
                        a.artist_name,
                        a.updated_on,
                        COUNT(DISTINCT t.track_id) as track_count,
                        COUNT(DISTINCT r.release_id) as release_count,
                        ROUND(AVG(af.danceability), 3) as avg_danceability,
                        ROUND(AVG(af.energy), 3) as avg_energy,
                        ROUND(AVG(af.valence), 3) as avg_valence
                    FROM artist a
                    LEFT JOIN artist_track at ON a.artist_id = at.artist_id
                    LEFT JOIN track t ON at.track_id = t.track_id
                    LEFT JOIN `release` r ON t.release_id = r.release_id
                    LEFT JOIN audio_features af ON t.isrc = af.isrc
                    WHERE 1=1
                    {search_condition}
                    GROUP BY a.artist_id, a.artist_name, a.updated_on
                    HAVING track_count >= :min_tracks
                )
                SELECT 
                    artist_id,
                    artist_name,
                    updated_on,
                    track_count,
                    release_count,
                    avg_danceability,
                    avg_energy,
                    avg_valence
                FROM artist_stats
                ORDER BY 
                    CASE 
                        WHEN :sort_by = 'name' THEN artist_name
                        WHEN :sort_by = 'tracks' THEN track_count
                        WHEN :sort_by = 'danceability' THEN avg_danceability
                        WHEN :sort_by = 'energy' THEN avg_energy
                        WHEN :sort_by = 'release_count' THEN release_count
                    END {order}
                LIMIT :limit OFFSET :offset
            """

            # Add search condition if search parameter is provided
            search_condition = "AND a.artist_name LIKE :search" if search else ""
            query = text(base_query.format(search_condition=search_condition, order=order))

            # Calculate offset
            offset = (page - 1) * per_page

            # Execute query with parameters
            params = {
                'min_tracks': min_tracks,
                'sort_by': sort_by,
                'limit': per_page,
                'offset': offset
            }
            if search:
                params['search'] = f'%{search}%'

            # Get total count for pagination
            count_query = text("""
            SELECT COUNT(*) FROM (
                SELECT a.artist_id
                FROM artist a
                LEFT JOIN artist_track at ON a.artist_id = at.artist_id
                WHERE 1=1 {search_condition}
                GROUP BY a.artist_id
                HAVING COUNT(at.track_id) >= :min_tracks
            ) as counted
        """.format(search_condition=search_condition))

            total = conn.execute(count_query, params).scalar()
            results = conn.execute(query, params).fetchall()

            artists = []
            for row in results:
                artists.append({
                    'artist_id': row[0],
                    'artist_name': row[1],
                    'updated_on': row[2],
                    'track_count': row[3],
                    'release_count': row[4],
                    'avg_danceability': row[5],
                    'avg_energy': row[6],
                    'avg_valence': row[7]
                })

            return jsonify({
                'total': total,
                'page': page,
                'per_page': per_page,
                'artists': artists
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/artists/<artist_id>', methods=['GET'])
def get_artist(artist_id):
    try:
        # Get optional query parameters
        include_audio_features = request.args.get('include_audio_features', 'false').lower() == 'true'
        sort_tracks_by = request.args.get('sort_tracks_by', 'release_date')  # release_date, title
        sort_order = request.args.get('sort_order', 'desc').upper()  # ASC or DESC
        
        with db.engine.connect() as conn:
            # Get artist details
            check_query = text("""
                SELECT 
                    artist_id, 
                    artist_name, 
                    updated_on,
                    (SELECT COUNT(*) FROM artist_track WHERE artist_id = :artist_id) as track_count,
                    (SELECT COUNT(DISTINCT release_id) 
                     FROM track t 
                     JOIN artist_track at ON t.track_id = at.track_id 
                     WHERE at.artist_id = :artist_id) as release_count
                FROM artist 
                WHERE artist_id = :artist_id
            """)
            
            result = conn.execute(check_query, {'artist_id': artist_id}).first()
            
            if not result:
                return jsonify({
                    'error': 'Artist not found',
                    'requested_id': artist_id
                }), 404
            
            # Build base response
            artist_data = {
                'artist_id': result[0],
                'artist_name': result[1],
                'updated_on': result[2].isoformat() if result[2] else None,
                'track_count': result[3],
                'release_count': result[4]
            }
            
            # Construct tracks query based on parameters
            tracks_query = f"""
                SELECT 
                    t.track_id,
                    t.track_title,
                    r.release_title,
                    DATE_FORMAT(r.release_date, '%Y-%m-%d') as release_date
                    {', af.danceability, af.energy, af.valence' if include_audio_features else ''}
                FROM track t
                JOIN artist_track at ON t.track_id = at.track_id
                JOIN `release` r ON t.release_id = r.release_id
                {' JOIN audio_features af ON t.isrc = af.isrc' if include_audio_features else ''}
                WHERE at.artist_id = :artist_id
                ORDER BY 
                    CASE 
                        WHEN :sort_by = 'release_date' THEN r.release_date
                        WHEN :sort_by = 'title' THEN t.track_title
                    END {sort_order}
            """
            
            tracks_result = conn.execute(
                text(tracks_query), 
                {
                    'artist_id': artist_id,
                    'sort_by': sort_tracks_by
                }
            ).fetchall()
            
            # Convert tracks to list of dictionaries
            tracks = []
            for track in tracks_result:
                track_dict = {
                    'track_id': track[0],
                    'track_title': track[1],
                    'release_title': track[2],
                    'release_date': track[3]
                }
                
                # Add audio features if requested
                if include_audio_features:
                    track_dict.update({
                        'danceability': track[4],
                        'energy': track[5],
                        'valence': track[6]
                    })
                
                tracks.append(track_dict)
            
            artist_data['tracks'] = tracks
            
            return jsonify(artist_data)
            
    except Exception as e:
        print(f"Error in get_artist: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@bp.route('/tracks', methods=['GET'])
def get_tracks():
    """
    Get list of tracks with filtering and sorting options.
    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 10)
    - sort_by: Sort field (title, release_date, danceability, energy) (default: release_date)
    - order: Sort order (asc, desc) (default: desc)
    - min_danceability: Minimum danceability score (0-1)
    - min_energy: Minimum energy score (0-1)
    - search: Search in track title
    - release_year: Filter by release year
    """
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        sort_by = request.args.get('sort_by', 'release_date')
        order = request.args.get('order', 'desc').upper()
        min_danceability = request.args.get('min_danceability', 0.0, type=float)
        min_energy = request.args.get('min_energy', 0.0, type=float)
        search = request.args.get('search', '')
        release_year = request.args.get('release_year', type=int)

        with db.engine.connect() as conn:
            # Build query conditions
            conditions = ["1=1"]
            params = {
                'min_danceability': min_danceability,
                'min_energy': min_energy,
                'limit': per_page,
                'offset': (page - 1) * per_page
            }

            if search:
                conditions.append("t.track_title LIKE :search")
                params['search'] = f'%{search}%'
            
            if release_year:
                conditions.append("YEAR(r.release_date) = :release_year")
                params['release_year'] = release_year

            if min_danceability > 0:
                conditions.append("af.danceability >= :min_danceability")
            
            if min_energy > 0:
                conditions.append("af.energy >= :min_energy")

            # Construct main query
            query = text(f"""
                SELECT 
                    t.track_id,
                    t.track_title,
                    r.release_title,
                    DATE_FORMAT(r.release_date, '%Y-%m-%d') as release_date,
                    GROUP_CONCAT(a.artist_name SEPARATOR ', ') as artists,
                    af.danceability,
                    af.energy,
                    af.valence,
                    af.tempo,
                    af.instrumentalness,
                    r.label_name
                FROM track t
                JOIN audio_features af ON t.isrc = af.isrc
                JOIN `release` r ON t.release_id = r.release_id
                JOIN artist_track at ON t.track_id = at.track_id
                JOIN artist a ON at.artist_id = a.artist_id
                WHERE {' AND '.join(conditions)}
                GROUP BY t.track_id, t.track_title, r.release_title, r.release_date,
                         af.danceability, af.energy, af.valence, af.tempo,
                         af.instrumentalness, r.label_name
                ORDER BY 
                    CASE 
                        WHEN :sort_by = 'title' THEN t.track_title
                        WHEN :sort_by = 'release_date' THEN r.release_date
                        WHEN :sort_by = 'danceability' THEN af.danceability
                        WHEN :sort_by = 'energy' THEN af.energy
                        WHEN :sort_by = 'valence' THEN af.valence
                        WHEN :sort_by = 'tempo' THEN af.tempo
                    END {order}
                LIMIT :limit OFFSET :offset
            """)

            # Get total count
            count_query = text(f"""
                SELECT COUNT(DISTINCT t.track_id)
                FROM track t
                JOIN audio_features af ON t.isrc = af.isrc
                JOIN `release` r ON t.release_id = r.release_id
                WHERE {' AND '.join(conditions)}
            """)

            params['sort_by'] = sort_by
            results = conn.execute(query, params).fetchall()
            total = conn.execute(count_query, params).scalar()

            tracks = []
            for row in results:
                tracks.append({
                    'track_id': row[0],
                    'track_title': row[1],
                    'release_title': row[2],
                    'release_date': row[3],
                    'artists': row[4].split(', '),
                    'audio_features': {
                        'danceability': row[5],
                        'energy': row[6],
                        'valence': row[7],
                        'tempo': row[8],
                        'instrumentalness': row[9]
                    },
                    'label': row[10]
                })

            return jsonify({
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': (total + per_page - 1) // per_page,
                'tracks': tracks,
                'filters': {
                    'min_danceability': min_danceability,
                    'min_energy': min_energy,
                    'search': search,
                    'release_year': release_year,
                    'sort_by': sort_by,
                    'order': order
                }
            })

    except Exception as e:
        print(f"Error in get_tracks: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@bp.route('/tracks/<track_id>', methods=['GET'])
def get_track(track_id):
    """Get detailed information about a specific track."""
    query = text("""
        SELECT 
            t.track_id,
            t.track_title,
            t.duration_ms,
            t.isrc,
            r.release_title,
            r.release_date,
            r.label_name,
            GROUP_CONCAT(a.artist_name) as artists,
            af.danceability,
            af.energy,
            af.valence,
            af.tempo,
            af.loudness,
            af.instrumentalness,
            af.acousticness,
            af.liveness,
            af.speechiness
        FROM track t
        JOIN audio_features af ON t.isrc = af.isrc
        JOIN `release` r ON t.release_id = r.release_id
        JOIN artist_track at ON t.track_id = at.track_id
        JOIN artist a ON at.artist_id = a.artist_id
        WHERE t.track_id = :track_id
        GROUP BY t.track_id, t.track_title, t.duration_ms, t.isrc,
                 r.release_title, r.release_date, r.label_name,
                 af.danceability, af.energy, af.valence, af.tempo,
                 af.loudness, af.instrumentalness, af.acousticness,
                 af.liveness, af.speechiness
    """)
    
    result = db.session.execute(query, {'track_id': track_id}).first()
    
    if not result:
        return jsonify({'error': 'Track not found'}), 404
    
    return jsonify(dict(result))