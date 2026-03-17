-- EverNear Cloud Backend — Initial Schema
-- Apply via Supabase SQL editor or CLI

-- Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    display_name TEXT DEFAULT '',
    preferred_name TEXT DEFAULT '',
    voice_preference TEXT DEFAULT '',
    onboarding_completed BOOLEAN DEFAULT FALSE,
    onboarding_state JSONB DEFAULT '{}',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Memories
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    category TEXT NOT NULL CHECK (category IN (
        'family','health','preferences','stories','emotions',
        'meaning','culture','faith','interests','caregivers','routine'
    )),
    content TEXT NOT NULL,
    source_turn_id UUID,
    importance INTEGER DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    turn_count INTEGER DEFAULT 0,
    summary TEXT DEFAULT ''
);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    audio_duration_ms INTEGER,
    latency_ms INTEGER,
    metrics JSONB DEFAULT '{}',
    sequence INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Consent Logs
CREATE TABLE IF NOT EXISTS consent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    consent_type TEXT NOT NULL CHECK (consent_type IN (
        'health_data','caregiver_sharing','recording','data_retention'
    )),
    granted BOOLEAN NOT NULL,
    disclosure_version TEXT DEFAULT '1.0',
    disclosure_hash TEXT DEFAULT '',
    method TEXT DEFAULT 'tap' CHECK (method IN ('voice','tap','onboarding')),
    ip_address TEXT DEFAULT '',
    device_info TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

-- Audit Log
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    action TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    ip_address TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Caregivers (schema only)
CREATE TABLE IF NOT EXISTS caregivers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT DEFAULT '',
    display_name TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- User-Caregiver Links (schema only)
CREATE TABLE IF NOT EXISTS user_caregivers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    caregiver_id UUID REFERENCES caregivers(id) ON DELETE CASCADE NOT NULL,
    relationship TEXT DEFAULT '',
    permissions JSONB DEFAULT '{}',
    authorized_at TIMESTAMPTZ DEFAULT now(),
    authorized_via TEXT DEFAULT 'onboarding',
    active BOOLEAN DEFAULT TRUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(user_id, category);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_consent_logs_user_id ON consent_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);

-- Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE consent_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- RLS Policies — user isolation
CREATE POLICY user_isolation_users ON users
    FOR ALL USING (id = auth.uid());

CREATE POLICY user_isolation_memories ON memories
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY user_isolation_conversations ON conversations
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY user_isolation_messages ON messages
    FOR ALL USING (
        conversation_id IN (
            SELECT id FROM conversations WHERE user_id = auth.uid()
        )
    );

CREATE POLICY user_isolation_consent ON consent_logs
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY user_isolation_audit ON audit_log
    FOR ALL USING (user_id = auth.uid());
