-- Migration 002: Add confidence column to memories table
-- Required by memory extraction pipeline (server/memory.py)
-- and validator memory cache (server/memory_cache.py)

ALTER TABLE memories ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.5;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS verified BOOLEAN DEFAULT FALSE;
