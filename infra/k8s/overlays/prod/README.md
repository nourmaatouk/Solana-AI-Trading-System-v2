# Production Environment

Kubernetes configuration for production environment.

## Purpose

Defines:
- Production resource allocations
- Production database configurations
- Production API endpoints
- Production security configurations
- Production monitoring and alerting

## Features

- High availability setup
- Auto-scaling policies
- Resource quotas and limits
- Network policies and security
- Backup and disaster recovery

## Usage

Deploy to prod cluster with:
```
kubectl apply -k infra/k8s/overlays/prod
```

**Note**: Requires prod credentials and approvals.
