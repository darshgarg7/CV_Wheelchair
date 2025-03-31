-- *****************************************************************************************************
-- Database Schema for Wheelchair Control Gesture Tracking
-- This script creates and ensures the necessary tables and relationships for storing gestures and logs.
-- The database tracks gestures and actions, it has built-in integrity and performance improvements.
-- *****************************************************************************************************

-- Step 1: Temporarily disable foreign key checks to safely alter tables if needed
PRAGMA foreign_keys = OFF;

-- Set SQLite performance optimizations
PRAGMA journal_mode = WAL;  -- Write-Ahead Logging
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;

-- Step 2: Create the 'gestures' table (if not already existing)
CREATE TABLE IF NOT EXISTS gestures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,            -- Unique identifier for each gesture
    name TEXT NOT NULL UNIQUE,                       -- Gesture name (e.g., 'Swipe Left', 'Swipe Up')
    command TEXT NOT NULL,                           -- Command associated with the gesture (e.g., 'Move Forward')
    coordinates TEXT NOT NULL                        -- Gesture coordinates (stored as JSON string for flexibility)
);

-- Step 3: Create the 'logs' table (if not already existing)
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,            -- Unique log entry ID
    gesture_id INTEGER,                              -- Foreign key referencing the gestures table
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,    -- Automatically sets the time when the action occurs
    action TEXT NOT NULL,                            -- Description of the action performed (e.g., 'Gesture Recognized')
    FOREIGN KEY (gesture_id) REFERENCES gestures(id) ON DELETE CASCADE  -- Ensures logs are deleted when related gesture is removed
);

-- Step 4: Improve performance by creating indexes
CREATE INDEX IF NOT EXISTS idx_gesture_name ON gestures(name);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_gesture_id ON logs(gesture_id);

-- Step 5: Re-enable foreign key checks to maintain database integrity
PRAGMA foreign_keys = ON;
