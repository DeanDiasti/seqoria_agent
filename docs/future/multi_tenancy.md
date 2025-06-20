# Multi-Tenancy Guide

!!! warning "Coming Soon"
    This feature is currently in the prototype phase. The documentation will be available once the API is stable.

This guide will explain how to build multi-tenant applications with xmcp. Key concepts will include:

-   Using a single `AIClient` instance to serve multiple tenants.
-   Isolating conversation history per tenant.
-   Injecting tenant context for logging and caching.
-   Implementing per-tenant quotas and usage tracking. 