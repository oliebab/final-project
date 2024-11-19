use diggerz;

-- Increase timeout and execution time limits
SET SESSION wait_timeout=28800;
SET SESSION interactive_timeout=28800;
SET SESSION max_execution_time=1000000;  -- 1000 seconds

-- Set a larger group concat max length
SET SESSION group_concat_max_len=1000000;

-- Disable strict mode temporarily if needed
SET SESSION sql_mode='';

select * from artist where artist_name = "alva noto";
select * from artist_track;
SET SESSION max_execution_time = 100000;

SELECT COUNT(*) FROM artist;
SELECT COUNT(*) FROM track;
SELECT COUNT(*) FROM `release`;
SELECT COUNT(*) FROM audio_features;
SELECT COUNT(*) FROM artist_track;

-- Top 10 artists in track count
SELECT 
    a.artist_name,
    COUNT(DISTINCT at.track_id) AS track_count
FROM artist a
JOIN artist_track at ON a.artist_id = at.artist_id
GROUP BY a.artist_id
ORDER BY track_count DESC
LIMIT 10;

-- Release Count by Year

CREATE TEMPORARY TABLE temp_release_tracks AS
SELECT 
    r.release_id,
    YEAR(r.release_date) AS release_year,
    t.track_id
FROM `release` r
JOIN track t ON r.release_id = t.release_id
WHERE r.release_date IS NOT NULL;

SELECT 
    release_year,
    COUNT(DISTINCT track_id) AS track_count
FROM temp_release_tracks
GROUP BY release_year
ORDER BY release_year DESC
LIMIT 10;

-- Label Analysis
SELECT 
    label_name,
    COUNT(*) as release_count
FROM `release`
WHERE label_name IS NOT NULL
GROUP BY label_name
ORDER BY release_count DESC
LIMIT 20;

-- Basic Audio Features Summary
SELECT 
    ROUND(AVG(danceability), 3) as avg_danceability,
    ROUND(AVG(energy), 3) as avg_energy,
    COUNT(*) as total_tracks
FROM audio_features
LIMIT 1;

-- Get alva noto tracks
SELECT a.artist_name, t.track_title
FROM artist a
JOIN artist_track at ON a.artist_id = at.artist_id
JOIN track t ON at.track_id = t.track_id
WHERE a.artist_id = '1zrqDVuh55auIRthalFdXp'
LIMIT 20;

-- 1. Top 10 Labels in 2023 by Count
-- Temporary table with 2023 releases
CREATE TEMPORARY TABLE temp_2023_releases AS
SELECT release_id, label_name
FROM `release`
WHERE YEAR(release_date) = 2023
    AND label_name IS NOT NULL
LIMIT 100000;

-- Add index to temporary table
ALTER TABLE temp_2023_releases ADD INDEX (release_id), ADD INDEX (label_name);

-- Query from temporary table
SELECT 
    label_name,
    COUNT(DISTINCT tr.release_id) as releases_count
FROM temp_2023_releases tr
GROUP BY label_name
HAVING releases_count >= 5
ORDER BY releases_count DESC
LIMIT 10;

-- 2. Audio Feature Analysis for 2023
-- Create temporary table for 2023 tracks with features
CREATE TEMPORARY TABLE temp_2023_tracks AS
SELECT 
    t.track_id,
    r.release_date,
    af.danceability,
    af.energy
FROM track t
JOIN `release` r ON t.release_id = r.release_id
JOIN audio_features af ON t.isrc = af.isrc
WHERE YEAR(r.release_date) = 2023
LIMIT 100000;

-- Add index
ALTER TABLE temp_2023_tracks ADD INDEX (release_date);

-- Query monthly averages
SELECT 
    MONTH(release_date) as release_month,
    COUNT(DISTINCT track_id) as tracks_count,
    ROUND(AVG(danceability), 3) as avg_danceability,
    ROUND(AVG(energy), 3) as avg_energy
FROM temp_2023_tracks
GROUP BY MONTH(release_date)
ORDER BY release_month;

-- 3. Music Type Categories based on Audio Features
CREATE TEMPORARY TABLE temp_music_categories AS
SELECT 
    t.track_id,
    t.track_title,
    r.release_date,
    af.danceability,
    af.energy,
    af.valence,
    CASE 
        WHEN af.danceability >= 0.7 AND af.energy >= 0.7 THEN 'Dance/Party'
        WHEN af.energy >= 0.7 AND af.valence <= 0.3 THEN 'Intense/Dark'
        WHEN af.energy <= 0.3 AND af.instrumentalness >= 0.7 THEN 'Ambient/Chill'
        WHEN af.acousticness >= 0.7 THEN 'Acoustic'
        WHEN af.energy >= 0.7 AND af.valence >= 0.7 THEN 'Upbeat/Happy'
        ELSE 'Other'
    END as music_category
FROM track t
JOIN `release` r ON t.release_id = r.release_id
JOIN audio_features af ON t.isrc = af.isrc
WHERE YEAR(r.release_date) >= 2023
LIMIT 100000;

-- Analysis of music categories
SELECT 
    music_category,
    COUNT(*) as track_count,
    ROUND(AVG(danceability), 3) as avg_danceability,
    ROUND(AVG(energy), 3) as avg_energy,
    ROUND(AVG(valence), 3) as avg_valence
FROM temp_music_categories
GROUP BY music_category
ORDER BY track_count DESC;

-- 4. Label Specializations
CREATE TEMPORARY TABLE temp_label_features AS
SELECT 
    r.label_name,
    af.danceability,
    af.energy,
    af.valence,
    af.instrumentalness,
    af.acousticness
FROM `release` r
JOIN track t ON t.release_id = r.release_id
JOIN audio_features af ON t.isrc = af.isrc
WHERE YEAR(r.release_date) >= 2023
    AND r.label_name IS NOT NULL
LIMIT 100000;

SELECT 
    label_name,
    COUNT(*) as track_count,
    CASE 
        WHEN AVG(danceability) >= 0.7 THEN 'Dance-focused'
        WHEN AVG(instrumentalness) >= 0.7 THEN 'Instrumental-focused'
        WHEN AVG(acousticness) >= 0.7 THEN 'Acoustic-focused'
        WHEN AVG(energy) >= 0.7 THEN 'High-energy'
        ELSE 'Mixed'
    END as label_specialty
FROM temp_label_features
GROUP BY label_name
HAVING track_count >= 10
ORDER BY track_count DESC
LIMIT 15;