ALTER TABLE public.screenings
  ADD COLUMN model_confidence_raw REAL CHECK (model_confidence_raw BETWEEN 0 AND 1),
  ADD COLUMN model_confidence_calibrated REAL CHECK (model_confidence_calibrated BETWEEN 0 AND 1),
  ADD COLUMN uncertainty_label TEXT CHECK (uncertainty_label IN ('low', 'medium', 'high', 'unavailable')),
  ADD COLUMN quality_model_version TEXT,
  ADD COLUMN classifier_model_version TEXT,
  ADD COLUMN segmentation_model_version TEXT,
  ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (review_status IN ('pending', 'reviewed', 'overridden', 'inconclusive')),
  ADD COLUMN reviewed_at TIMESTAMPTZ,
  ADD COLUMN processing_mode TEXT NOT NULL DEFAULT 'cloud'
    CHECK (processing_mode IN ('local', 'cloud', 'auto'));

CREATE TABLE public.screening_reviews (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  screening_id UUID NOT NULL REFERENCES public.screenings(id) ON DELETE CASCADE,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  reviewer_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
  decision TEXT NOT NULL CHECK (decision IN ('reviewed', 'overridden', 'inconclusive')),
  reviewer_grade SMALLINT CHECK (reviewer_grade BETWEEN 0 AND 4),
  notes TEXT CHECK (char_length(notes) <= 2000),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX screening_reviews_owner_created_idx
  ON public.screening_reviews (owner_id, created_at DESC);
CREATE INDEX screening_reviews_screening_idx
  ON public.screening_reviews (screening_id);
CREATE INDEX screening_reviews_reviewer_idx
  ON public.screening_reviews (reviewer_id);

ALTER TABLE public.screening_reviews ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.owns_screening(screening_uuid UUID, owner_uuid UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.screenings
    WHERE id = screening_uuid
      AND user_id = owner_uuid
      AND owner_uuid = (SELECT auth.uid())
  );
$$;

REVOKE ALL ON FUNCTION public.owns_screening(UUID, UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.owns_screening(UUID, UUID) TO authenticated;

CREATE POLICY screening_reviews_owner_select ON public.screening_reviews
  FOR SELECT TO authenticated
  USING (owner_id = (SELECT auth.uid()));

CREATE POLICY screening_reviews_owner_insert ON public.screening_reviews
  FOR INSERT TO authenticated
  WITH CHECK (
    owner_id = (SELECT auth.uid())
    AND reviewer_id = (SELECT auth.uid())
    AND public.owns_screening(screening_id, owner_id)
  );

CREATE OR REPLACE FUNCTION public.apply_screening_review()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
BEGIN
  UPDATE public.screenings
  SET review_status = NEW.decision,
      reviewed_at = NEW.created_at
  WHERE id = NEW.screening_id
    AND user_id = NEW.owner_id;
  RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.apply_screening_review() FROM PUBLIC;

CREATE TRIGGER screening_reviews_apply_status
AFTER INSERT ON public.screening_reviews
FOR EACH ROW EXECUTE FUNCTION public.apply_screening_review();

GRANT SELECT, INSERT ON public.screening_reviews TO authenticated;
REVOKE UPDATE, DELETE ON public.screening_reviews FROM anon, authenticated;

REVOKE UPDATE ON public.screenings FROM anon, authenticated;
GRANT UPDATE (status) ON public.screenings TO authenticated;
