# Stage AE human-review handling checklist

Status: `PENDING_PI_REVIEW_BEFORE_DISTRIBUTION`

- [ ] PI confirms that the intended reviewers and local institutional process
      permit review of these de-identified simulation clips.
- [ ] Reviewers receive only an HR ID and the neutral package; no personal
      information is written to Git.
- [ ] Reviewers are told that this is endpoint observability, not a request to
      diagnose model safety or certify a physical failure.
- [ ] Reviewers can stop if a clip is disturbing, inaccessible, or cannot be
      judged under the rubric.
- [ ] Returned CSVs are transferred through the agreed private channel and are
      sealed before any hidden mapping is opened.
- [ ] The repository retains the distinction between `human_review_completed`
      and `human_endpoint_confirmed`.
