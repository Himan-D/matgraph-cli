# MatGraph Documentation

## Table of Contents

1. [Installation](installation.md) — including the 60-second no-key demo
2. [Tutorials](tutorials.md)
3. [CLI Reference](cli-reference.md)
4. [Python SDK](sdk-reference.md)
5. [GraphQL API](graphql-api.md)
6. [Models](models.md)
7. [MatGraph vs raw stack](comparison.md)
8. [Growth & Distribution](growth.md) — maintainer runbook for growing adoption

---

## Overview

MatGraph is a Deep Learning toolkit for Materials Science. It provides three interfaces to the same underlying prediction engine:

| Interface | Best for |
|---|---|
| CLI (`matgraph`) | Quick predictions, scripting, automation |
| Python SDK (`MatGraphSDK`) | Jupyter Notebooks, ML pipelines, custom scripts |
| GraphQL API | Web applications, microservices, dashboards |

All three interfaces share the same models (M3GNet, MEGNet, CGCNN, CHGNet), the same Materials Project data source, and the same SQLite cache. New here? Start with [Installation](installation.md) — the `--demo` mode gives you a first successful command in under a minute, no API key required.

---

## Quick Links

- [PyPI Package](https://pypi.org/project/matgraph-cli/)
- [GitHub Repository](https://github.com/Himan-D/matgraph-cli)
- [Issue Tracker](https://github.com/Himan-D/matgraph-cli/issues)
- [Contributing Guide](https://github.com/Himan-D/matgraph-cli/blob/main/CONTRIBUTING.md)
