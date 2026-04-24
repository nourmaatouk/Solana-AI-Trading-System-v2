# Development Environment

Kubernetes configuration for development environment.

## Purpose

Defines:
- Development resource allocations
- Dev database configurations
- Dev API endpoints
- Dev service accounts
- Dev monitoring and logging

## Usage

Deploy to dev cluster with:
```
kubectl apply -k infra/k8s/overlays/dev
```
