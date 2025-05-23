-- Initialize Name Matching Database
-- This script creates the necessary tables and indexes for the name matching application

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS namematching CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE namematching;

-- Create person records table
CREATE TABLE IF NOT EXISTS person_records (
    hh_id BIGINT UNSIGNED PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    middle_name_last_name VARCHAR(200),
    birthdate DATE,
    province_name VARCHAR(100),
    city_name VARCHAR(100),
    barangay_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_names (first_name, middle_name_last_name),
    INDEX idx_birthdate (birthdate),
    INDEX idx_geography (province_name, city_name, barangay_name),
    INDEX idx_created_at (created_at),
    
    -- Full-text search indexes
    FULLTEXT INDEX ft_first_name (first_name) WITH PARSER ngram,
    FULLTEXT INDEX ft_middle_last_name (middle_name_last_name) WITH PARSER ngram
) ENGINE=InnoDB;

-- Create match results table
CREATE TABLE IF NOT EXISTS match_results (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    record1_id BIGINT UNSIGNED NOT NULL,
    record2_id BIGINT UNSIGNED NOT NULL,
    similarity_score DECIMAL(5,4) NOT NULL,
    classification ENUM('match', 'non_match', 'manual_review') NOT NULL,
    component_scores JSON,
    processing_time_ms DECIMAL(10,3),
    algorithm_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_record1 (record1_id),
    INDEX idx_record2 (record2_id),
    INDEX idx_score (similarity_score),
    INDEX idx_classification (classification),
    INDEX idx_created_at (created_at),
    
    -- Unique constraint to prevent duplicate matches
    UNIQUE KEY unique_match (record1_id, record2_id),
    
    -- Foreign key constraints
    FOREIGN KEY (record1_id) REFERENCES person_records(hh_id) ON DELETE CASCADE,
    FOREIGN KEY (record2_id) REFERENCES person_records(hh_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Create blocking keys table for performance optimization
CREATE TABLE IF NOT EXISTS blocking_keys (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    record_id BIGINT UNSIGNED NOT NULL,
    key_type ENUM('soundex', 'first_char', 'ngram', 'geographic') NOT NULL,
    key_value VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_record_id (record_id),
    INDEX idx_key_type_value (key_type, key_value),
    INDEX idx_key_value (key_value),
    
    -- Foreign key constraint
    FOREIGN KEY (record_id) REFERENCES person_records(hh_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id BIGINT UNSIGNED,
    action ENUM('INSERT', 'UPDATE', 'DELETE') NOT NULL,
    old_values JSON,
    new_values JSON,
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_table_record (table_name, record_id),
    INDEX idx_action (action),
    INDEX idx_timestamp (timestamp),
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB;

-- Create configuration table for runtime settings
CREATE TABLE IF NOT EXISTS app_config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT,
    config_type ENUM('string', 'integer', 'float', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_config_key (config_key),
    INDEX idx_config_type (config_type)
) ENGINE=InnoDB;

-- Insert default configuration values
INSERT INTO app_config (config_key, config_value, config_type, description) VALUES
('match_threshold', '0.75', 'float', 'Threshold for classifying pairs as matches'),
('non_match_threshold', '0.55', 'float', 'Threshold for classifying pairs as non-matches'),
('enable_gpu', 'false', 'boolean', 'Enable GPU acceleration for matching'),
('enable_caching', 'true', 'boolean', 'Enable result caching'),
('cache_ttl', '3600', 'integer', 'Cache time-to-live in seconds'),
('max_batch_size', '10000', 'integer', 'Maximum batch size for processing'),
('component_weights', '{"first_name": 0.3, "middle_name": 0.2, "last_name": 0.3, "birthdate": 0.1, "geography": 0.1}', 'json', 'Weights for different name components')
ON DUPLICATE KEY UPDATE 
    config_value = VALUES(config_value),
    updated_at = CURRENT_TIMESTAMP;

-- Create sample data for testing (optional)
INSERT IGNORE INTO person_records (hh_id, first_name, middle_name_last_name, birthdate, province_name, city_name, barangay_name) VALUES
(1, 'Juan', 'dela Cruz', '1990-01-15', 'Metro Manila', 'Manila', 'Ermita'),
(2, 'Juan', 'de la Cruz', '1990-01-15', 'Metro Manila', 'Manila', 'Ermita'),
(3, 'Maria', 'Santos Garcia', '1985-03-22', 'Cebu', 'Cebu City', 'Lahug'),
(4, 'Jose', 'Rizal Mercado', '1988-06-19', 'Laguna', 'Calamba', 'Real'),
(5, 'Ana', 'Reyes Lopez', '1992-11-08', 'Davao del Sur', 'Davao City', 'Poblacion');

-- Create views for common queries
CREATE OR REPLACE VIEW recent_matches AS
SELECT 
    mr.id,
    mr.record1_id,
    mr.record2_id,
    pr1.first_name AS name1_first,
    pr1.middle_name_last_name AS name1_middle_last,
    pr2.first_name AS name2_first,
    pr2.middle_name_last_name AS name2_middle_last,
    mr.similarity_score,
    mr.classification,
    mr.created_at
FROM match_results mr
JOIN person_records pr1 ON mr.record1_id = pr1.hh_id
JOIN person_records pr2 ON mr.record2_id = pr2.hh_id
WHERE mr.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY mr.created_at DESC;

CREATE OR REPLACE VIEW high_confidence_matches AS
SELECT 
    mr.id,
    mr.record1_id,
    mr.record2_id,
    pr1.first_name AS name1_first,
    pr1.middle_name_last_name AS name1_middle_last,
    pr2.first_name AS name2_first,
    pr2.middle_name_last_name AS name2_middle_last,
    mr.similarity_score,
    mr.classification,
    mr.created_at
FROM match_results mr
JOIN person_records pr1 ON mr.record1_id = pr1.hh_id
JOIN person_records pr2 ON mr.record2_id = pr2.hh_id
WHERE mr.similarity_score >= 0.9 AND mr.classification = 'match'
ORDER BY mr.similarity_score DESC;

-- Create stored procedures for common operations
DELIMITER //

CREATE PROCEDURE GetMatchingStatistics()
BEGIN
    SELECT 
        COUNT(*) as total_matches,
        AVG(similarity_score) as avg_score,
        COUNT(CASE WHEN classification = 'match' THEN 1 END) as confirmed_matches,
        COUNT(CASE WHEN classification = 'non_match' THEN 1 END) as non_matches,
        COUNT(CASE WHEN classification = 'manual_review' THEN 1 END) as manual_reviews,
        MIN(created_at) as first_match,
        MAX(created_at) as last_match
    FROM match_results;
END //

CREATE PROCEDURE CleanupOldMatches(IN days_old INT)
BEGIN
    DELETE FROM match_results 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL days_old DAY);
    
    SELECT ROW_COUNT() as deleted_records;
END //

CREATE PROCEDURE GetPersonMatches(IN person_id BIGINT)
BEGIN
    SELECT 
        mr.id,
        CASE 
            WHEN mr.record1_id = person_id THEN mr.record2_id 
            ELSE mr.record1_id 
        END as matched_person_id,
        CASE 
            WHEN mr.record1_id = person_id THEN CONCAT(pr2.first_name, ' ', pr2.middle_name_last_name)
            ELSE CONCAT(pr1.first_name, ' ', pr1.middle_name_last_name)
        END as matched_person_name,
        mr.similarity_score,
        mr.classification,
        mr.created_at
    FROM match_results mr
    LEFT JOIN person_records pr1 ON mr.record1_id = pr1.hh_id
    LEFT JOIN person_records pr2 ON mr.record2_id = pr2.hh_id
    WHERE mr.record1_id = person_id OR mr.record2_id = person_id
    ORDER BY mr.similarity_score DESC;
END //

DELIMITER ;

-- Grant permissions to the application user
GRANT SELECT, INSERT, UPDATE, DELETE ON namematching.* TO 'nameuser'@'%';
GRANT EXECUTE ON namematching.* TO 'nameuser'@'%';

-- Optimize tables
OPTIMIZE TABLE person_records, match_results, blocking_keys, audit_log, app_config;

-- Show table information
SHOW TABLE STATUS FROM namematching;
