# Kubernetes Base Configuration

Base Kubernetes manifests shared across all environments.

## Purpose

Contains:
- Deployment templates
- Service definitions
- ConfigMap templates
- RBAC configurations
- Network policies

## Usage

Base configuration is extended by environment-specific overlays using Kustomize.
