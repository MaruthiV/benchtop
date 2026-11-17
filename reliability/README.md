# Cached server exercise

Two bounded runs each used 1,000 simulated clients and made 8,000 requests against the actual two-port server on one laptop. Both recorded zero unexpected errors, 2,000 receipt-only submissions and 1,000 successful capability-revocation checks. Neither used a model or a live lab worker.

The burst run was slow: tool-call p99 was about 132 seconds, including waiting for the HTTP client's connection pool. A follow-up limited client admission to 32 requests at a time. Its tool round-trip p99 was about 1.55 seconds, with admission wait p99 of 14.26 seconds recorded separately.

These timing definitions differ. The two numbers are not a measured server speedup. Client and server ran on the same laptop, HTTPX overhead was not isolated, and each setup was run once. The tests establish the observed receipt, isolation and revocation behavior under these particular cached workloads. They do not establish a production latency guarantee, a general error-rate bound, or 1,000 simultaneously running research agents.

The JSON reports include per-route request counts, timings, checks and the source hashes for the tested server. Only sanitized reports are included; temporary credentials and server logs remain outside this package.
