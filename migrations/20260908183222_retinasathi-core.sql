CREATE TABLE public.screenings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  patient_code TEXT NOT NULL CHECK (char_length(patient_code) BETWEEN 1 AND 32),
  patient_age SMALLINT CHECK (patient_age BETWEEN 1 AND 120),
  patient_sex TEXT CHECK (patient_sex IN ('female', 'male', 'other', 'unknown')),
  image_url TEXT NOT NULL,
  image_key TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('queued', 'processing', 'completed', 'needs_retake', 'failed')),
  quality_score REAL CHECK (quality_score BETWEEN 0 AND 1),
  quality_label TEXT CHECK (quality_label IN ('good', 'usable', 'poor')),
  dr_grade SMALLINT CHECK (dr_grade BETWEEN 0 AND 4),
  dme_risk SMALLINT CHECK (dme_risk BETWEEN 0 AND 2),
  confidence REAL CHECK (confidence BETWEEN 0 AND 1),
  referable_dr BOOLEAN,
  recommendation TEXT,
  model_version TEXT NOT NULL DEFAULT 'idrid-mobile-v0.1',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, image_key)
);

CREATE TABLE public.screening_lesions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  screening_id UUID NOT NULL REFERENCES public.screenings(id) ON DELETE CASCADE,
  user_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  lesion_type TEXT NOT NULL CHECK (lesion_type IN ('microaneurysm', 'hemorrhage', 'hard_exudate', 'soft_exudate')),
  probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
  center_x REAL NOT NULL CHECK (center_x BETWEEN 0 AND 1),
  center_y REAL NOT NULL CHECK (center_y BETWEEN 0 AND 1),
  radius REAL NOT NULL CHECK (radius BETWEEN 0 AND 1),
  evidence TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX screenings_user_created_idx ON public.screenings (user_id, created_at DESC);
CREATE INDEX screening_lesions_user_idx ON public.screening_lesions (user_id);
CREATE INDEX screening_lesions_screening_idx ON public.screening_lesions (screening_id);

CREATE TRIGGER screenings_updated_at
  BEFORE UPDATE ON public.screenings
  FOR EACH ROW
  EXECUTE FUNCTION system.update_updated_at();

ALTER TABLE public.screenings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.screening_lesions ENABLE ROW LEVEL SECURITY;

CREATE POLICY screenings_owner_select ON public.screenings
  FOR SELECT TO authenticated
  USING (user_id = (SELECT auth.uid()));

CREATE POLICY screenings_owner_insert ON public.screenings
  FOR INSERT TO authenticated
  WITH CHECK (user_id = (SELECT auth.uid()));

CREATE POLICY screenings_owner_update ON public.screenings
  FOR UPDATE TO authenticated
  USING (user_id = (SELECT auth.uid()))
  WITH CHECK (user_id = (SELECT auth.uid()));

CREATE POLICY screenings_owner_delete ON public.screenings
  FOR DELETE TO authenticated
  USING (user_id = (SELECT auth.uid()));

CREATE POLICY lesions_owner_select ON public.screening_lesions
  FOR SELECT TO authenticated
  USING (user_id = (SELECT auth.uid()));

CREATE POLICY lesions_owner_insert ON public.screening_lesions
  FOR INSERT TO authenticated
  WITH CHECK (
    user_id = (SELECT auth.uid())
    AND EXISTS (
      SELECT 1 FROM public.screenings
      WHERE public.screenings.id = screening_id
        AND public.screenings.user_id = (SELECT auth.uid())
    )
  );

CREATE POLICY lesions_owner_delete ON public.screening_lesions
  FOR DELETE TO authenticated
  USING (user_id = (SELECT auth.uid()));

GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.screenings TO authenticated;
GRANT SELECT, INSERT, DELETE ON public.screening_lesions TO authenticated;

ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

CREATE POLICY retinal_screenings_owner_select ON storage.objects
  FOR SELECT TO authenticated
  USING (
    bucket = 'retinal-screenings'
    AND uploaded_by = (SELECT auth.jwt() ->> 'sub')
  );

CREATE POLICY retinal_screenings_owner_insert ON storage.objects
  FOR INSERT TO authenticated
  WITH CHECK (
    bucket = 'retinal-screenings'
    AND uploaded_by = (SELECT auth.jwt() ->> 'sub')
  );

CREATE POLICY retinal_screenings_owner_update ON storage.objects
  FOR UPDATE TO authenticated
  USING (
    bucket = 'retinal-screenings'
    AND uploaded_by = (SELECT auth.jwt() ->> 'sub')
  )
  WITH CHECK (
    bucket = 'retinal-screenings'
    AND uploaded_by = (SELECT auth.jwt() ->> 'sub')
  );

CREATE POLICY retinal_screenings_owner_delete ON storage.objects
  FOR DELETE TO authenticated
  USING (
    bucket = 'retinal-screenings'
    AND uploaded_by = (SELECT auth.jwt() ->> 'sub')
  );

GRANT USAGE ON SCHEMA storage TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON storage.objects TO authenticated;
