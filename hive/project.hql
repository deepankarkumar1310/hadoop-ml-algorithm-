-- Stage 11: Hive integration layer.
--
--   hive --hivevar start=1991 --hivevar end=2020 -f hive/project.hql
--
-- External tables sit over files already in HDFS, so dropping a table never
-- deletes data. Everything joins on censuscode, the all-India Census 2011
-- district code (1-640). Same logic as scripts/build_climate_baseline.py and
-- scripts/join_climate.py (the pandas fallback); the two paths are checked
-- against each other in the notebook.

CREATE DATABASE IF NOT EXISTS project;
USE project;

-- ---------- MapReduce output: one row per district-year ----------
DROP TABLE IF EXISTS district_year;
CREATE EXTERNAL TABLE district_year (
  censuscode INT, year INT, n_days INT, n_rain INT, n_tmax INT, n_tmin INT,
  annual_rain_mm DOUBLE, heavy_rain_days INT, rx5day_mm DOUBLE, max_dry_spell INT,
  mean_tmax DOUBLE, hot_days INT, tropical_nights INT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/project/processed/district_year';

-- ---------- socio-economic tables (header row skipped) ----------
DROP TABLE IF EXISTS census_2011;
CREATE EXTERNAL TABLE census_2011 (
  censuscode INT, state_code STRING, district_name STRING, total_population BIGINT,
  female_population BIGINT, literates BIGINT, female_literates BIGINT, total_workers BIGINT,
  literacy_rate DOUBLE, female_literacy_rate DOUBLE, agri_worker_share DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/project/processed/census_2011'
TBLPROPERTIES ('skip.header.line.count'='1');

DROP TABLE IF EXISTS ndap_crosswalk;
CREATE EXTERNAL TABLE ndap_crosswalk (
  district_code INT, district_name STRING, state_name STRING, censuscode INT,
  matched_census_name STRING, match_type STRING, match_score DOUBLE, climate_vul_in DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/project/processed/ndap_crosswalk'
TBLPROPERTIES ('skip.header.line.count'='1');

DROP TABLE IF EXISTS socioeconomic;
CREATE EXTERNAL TABLE socioeconomic (
  censuscode INT, district_name STRING, state_name STRING, climate_vul_in DOUBLE,
  literacy_rate DOUBLE, female_literacy_rate DOUBLE, agri_worker_share DOUBLE,
  mgnrega_days DOUBLE, yield_variability DOUBLE,
  z_literacy_rate DOUBLE, z_female_literacy_rate DOUBLE, z_agri_worker_share DOUBLE,
  z_mgnrega_days DOUBLE, z_yield_variability DOUBLE,
  development_score DOUBLE, dev_group STRING, vul_group STRING, quadrant STRING,
  match_type STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/project/processed/socioeconomic'
TBLPROPERTIES ('skip.header.line.count'='1');

-- ---------- join audit: how many districts survive each join ----------
SELECT 'census districts'            AS stage, COUNT(*) AS districts FROM census_2011
UNION ALL
SELECT 'ndap rows matched to census', COUNT(*) FROM ndap_crosswalk WHERE censuscode IS NOT NULL
UNION ALL
SELECT 'matched and present in census', COUNT(*)
  FROM ndap_crosswalk x JOIN census_2011 c ON x.censuscode = c.censuscode
UNION ALL
SELECT 'analysis set (complete cases)', COUNT(*) FROM socioeconomic
UNION ALL
SELECT 'districts with climate years', COUNT(DISTINCT censuscode) FROM district_year;

-- ---------- climate baseline: one profile per district ----------
-- AVG ignores NULLs, so an incomplete year drops out of that index only.
DROP TABLE IF EXISTS climate_baseline;
CREATE TABLE climate_baseline
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
AS
SELECT censuscode,
       ROUND(AVG(annual_rain_mm), 3)          AS annual_rain_mm,
       ROUND(STDDEV_SAMP(annual_rain_mm), 3)  AS rain_variability,
       ROUND(AVG(rx5day_mm), 3)               AS rx5day_mm,
       ROUND(AVG(heavy_rain_days), 3)         AS heavy_rain_days,
       ROUND(AVG(max_dry_spell), 3)           AS max_dry_spell,
       ROUND(AVG(mean_tmax), 3)               AS mean_tmax,
       ROUND(AVG(hot_days), 3)                AS hot_days,
       ROUND(AVG(tropical_nights), 3)         AS tropical_nights,
       COUNT(annual_rain_mm)                  AS n_years_rain,
       COUNT(mean_tmax)                       AS n_years_temp
FROM district_year
WHERE year BETWEEN ${hivevar:start} AND ${hivevar:end}
GROUP BY censuscode;

-- ---------- final integrated table: outcome + development + hazard ----------
DROP TABLE IF EXISTS integrated_district;
CREATE TABLE integrated_district
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
AS
SELECT s.*,
       b.annual_rain_mm, b.rain_variability, b.rx5day_mm, b.heavy_rain_days,
       b.max_dry_spell, b.mean_tmax, b.hot_days, b.tropical_nights,
       b.n_years_rain, b.n_years_temp
FROM socioeconomic s
JOIN climate_baseline b ON s.censuscode = b.censuscode;

SELECT quadrant, COUNT(*) AS districts,
       ROUND(AVG(annual_rain_mm), 1) AS rain_mm, ROUND(AVG(rx5day_mm), 1) AS rx5day,
       ROUND(AVG(hot_days), 1) AS hot_days, ROUND(AVG(max_dry_spell), 1) AS dry_spell
FROM integrated_district
GROUP BY quadrant
ORDER BY quadrant;
