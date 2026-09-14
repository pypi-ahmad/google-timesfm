"""Local, durable application services for the TimesFM-3 workbench.

Package root; exports nothing itself. Start reading at api.py (the HTTP
surface), which calls into services.py (execution logic) and store.py
(the durable job/record ledger) to follow the request-to-result flow.
"""
