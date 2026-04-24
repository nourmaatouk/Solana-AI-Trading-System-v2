# Kubernetes

Kubernetes manifests for orchestrating containerized services.

## Structure

- **base/** - Base configuration for all environments
- **overlays/** - Environment-specific overlays (dev, prod)

## Purpose

Defines:
- Deployment specifications
- Service definitions
- ConfigMaps and Secrets
- Persistent volumes
- Network policies
- Ingress configurations

## Deployment

Uses Kustomize for managing configurations across environments.

## Key Resources

- API deployments
- Agent service deployments
- Database statefulsets
- Monitoring and logging infrastructure
- Network and ingress configuration
