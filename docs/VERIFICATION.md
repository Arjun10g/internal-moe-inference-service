# Verification record

`artifacts/verification/verification.json` is the checked record from the most
recent final validation. Update it only after running the documented checks. It
records versions, pass/fail results, the tiny model architecture/parameter
count, generated token count, streaming completion, resident load count, GCP
pointer reachability, and environmental limitations such as an unavailable
Docker daemon.

Never copy a generated test checkpoint into the repository or production image. Do not treat the tiny random checkpoint as evidence of target-model quality or ~30B compatibility.
