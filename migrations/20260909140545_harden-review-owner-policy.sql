DROP POLICY IF EXISTS screening_reviews_owner_insert ON public.screening_reviews;

CREATE POLICY screening_reviews_owner_insert ON public.screening_reviews
  FOR INSERT TO authenticated
  WITH CHECK (
    owner_id = (SELECT auth.uid())
    AND reviewer_id = (SELECT auth.uid())
    AND EXISTS (
      SELECT 1
      FROM public.screenings
      WHERE public.screenings.id = screening_id
        AND public.screenings.user_id = (SELECT auth.uid())
    )
  );

REVOKE EXECUTE ON FUNCTION public.owns_screening(UUID, UUID) FROM authenticated;
REVOKE EXECUTE ON FUNCTION public.owns_screening(UUID, UUID) FROM PUBLIC;
DROP FUNCTION public.owns_screening(UUID, UUID);
